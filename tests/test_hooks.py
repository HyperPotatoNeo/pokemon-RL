"""Tests for the MultiTurnEnv hooks path — the interface verifiers will use.

Tests setup_state → get_prompt_messages → add_trajectory_step →
render_completion for both single and self-play modes, using mock
BattleManagers that ENFORCE the API contract.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    - Strict mocks ASSERT on contract violations (no silent deadlocks)
    - Tests verify values are the correct TYPE (not tuples where Battles expected)
    - Both positive and negative cases are checked
    - Every trajectory field is verified, not just length > 0

BUGS FIXED (these tests now pass):
    1. setup_state unpacks (idx, battle) tuples into _AgentContext objects
    2. _advance_selfplay buffers pending states, calls get_pending only after all acted
    3. _pending_states consumed correctly — P2 state no longer lost
    4. selfplay won=None gives symmetric 0.0 rewards to both players

Phase 4 renames applied:
    - opponent_mode="heuristic" → play_mode="single"
    - opponent_mode="self_play" → play_mode="self_play"
    - state["turn"] → state["game_turn"]
    - state["current_player"] → state["_current_agent_idx"]
    - state["battle"] → state["_agents"][idx].battle
    - step["player_idx"] → step["extras"]["agent_idx"]
    - step["parsed_action"] → step["extras"]["parsed_action"]
    - step["force_switch"] → step["extras"]["force_switch"]
    - _assign_rewards(trajectory, won) → _assign_rewards(state)
    - Completions use Messages format (list of dicts)
"""

import pytest
from unittest.mock import patch

from pokemon_rl.env import PokemonBattleEnv, _AgentContext


# ---- Mock infrastructure ----


class MockMove:
    """Minimal move mock."""
    def __init__(self, move_id="tackle", base_power=40):
        self.id = move_id
        self.base_power = base_power
        self.type = "normal"


class MockBattle:
    """Minimal Battle mock with enough attributes for env hooks.

    CRITICAL: This is a plain object, NOT a tuple. Tests verify that
    agent.battle contains MockBattle instances, not (idx, MockBattle) tuples.
    """
    def __init__(self, name="mock", turn=1, moves=None, switches=None):
        self.name = name
        self.turn = turn
        self.available_moves = moves if moves is not None else [MockMove()]
        self.available_switches = switches if switches is not None else []
        self.force_switch = False
        self.won = None
        self.battle_tag = f"mock-{name}"


class MockAction:
    """Minimal BattleOrder mock."""
    def __init__(self, msg="tackle"):
        self.message = f"/choose move {msg}"


class MockTranslator:
    """Mock StateTranslator that validates input types and supports Messages format.

    battle_to_prompt ASSERTS it receives a Battle-like object, not a tuple.
    This catches the setup_state unpacking bug immediately with a clear
    message rather than a confusing downstream error.
    """
    def battle_to_prompt(self, battle):
        assert not isinstance(battle, tuple), (
            f"battle_to_prompt received a tuple {battle!r} instead of a Battle "
            f"object. This means setup_state stored (player_idx, battle) tuples "
            f"from start_battle_selfplay instead of unpacking them."
        )
        return [
            {"role": "system", "content": "You are a Pokemon battle AI."},
            {"role": "user", "content": f"Battle state: {getattr(battle, 'name', '?')}"},
        ]

    def parse_action(self, text, battle):
        if isinstance(text, list):
            # Messages format: extract last assistant content
            for msg in reversed(text):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            else:
                return None
        if "move" in str(text).lower():
            return MockAction("parsed_move")
        return None

    def get_fallback_action(self, battle):
        return MockAction("fallback")

    @staticmethod
    def extract_completion_text(messages):
        """Convert Messages to string (Phase 4 addition)."""
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return str(messages)

    def extract_user_content(self, messages):
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return ""


class StrictMockHeuristicManager:
    """Mock BattleManager for single mode that enforces the API contract.

    Contract:
    1. start_battle() called once -> returns Battle
    2. step(action) called per turn -> returns (Battle, False) or (None, True)
    3. get_result() called after finish
    """
    def __init__(self, game_turns=3):
        self._turns = game_turns
        self._current = 0
        self._step_count = 0
        self._started = False
        self._finished = False

    async def start_battle(self, **kwargs):
        assert not self._started, "start_battle called twice"
        self._started = True
        self._current = 1
        return MockBattle("turn1", turn=1)

    async def step(self, action):
        assert self._started, "step called before start_battle"
        assert not self._finished, "step called after game over"
        self._step_count += 1
        self._current += 1
        if self._current > self._turns:
            self._finished = True
            return None, True
        return MockBattle(f"turn{self._current}", turn=self._current), False

    def get_result(self):
        assert self._finished, "get_result before finish"
        return {
            "won": True, "turns": self._current, "steps": self._step_count,
            "format": "gen1randombattle", "battle_tag": "mock-heuristic",
            "selfplay": False,
        }

    async def close(self):
        self._finished = True

    @property
    def is_finished(self):
        return self._finished


class StrictMockSelfplayManager:
    """Mock BattleManager that STRICTLY enforces the selfplay API contract.

    THE CONTRACT:
    1. start_battle_selfplay() returns list of (player_idx, battle) tuples
    2. For EACH tuple, caller MUST call submit_selfplay_action(player_idx, action)
    3. ONLY AFTER ALL actions submitted, call get_pending_selfplay_states()
    4. Repeat from step 2 until get_pending returns []

    If step 3 is violated (get_pending before all actions), raises
    AssertionError with a detailed diagnostic. This catches the bug
    without deadlocking.
    """
    def __init__(self, game_script=None):
        """
        Args:
            game_script: List of turns. Each turn is a list of (idx, MockBattle).
                Default: 3 turns of [(0, b), (1, b)], then game over.
        """
        if game_script is None:
            game_script = [
                [(0, MockBattle(f"p1_t{t}", turn=t)),
                 (1, MockBattle(f"p2_t{t}", turn=t))]
                for t in range(1, 4)
            ]
        self._script = game_script
        self._turn_idx = -1
        self._expected = set()
        self._received = set()
        self._step_count = 0
        self._finished = False

    async def start_battle_selfplay(self, **kwargs):
        self._turn_idx = 0
        turn = self._script[0]
        self._expected = {idx for idx, _ in turn}
        self._received = set()
        return list(turn)

    async def submit_selfplay_action(self, player_idx, action):
        assert player_idx in self._expected, (
            f"Unexpected action for player {player_idx}. "
            f"Expected: {self._expected}"
        )
        assert player_idx not in self._received, (
            f"Duplicate action for player {player_idx}"
        )
        self._received.add(player_idx)
        self._step_count += 1

    async def get_pending_selfplay_states(self):
        missing = self._expected - self._received
        assert not missing, (
            f"get_pending_selfplay_states called before all actions submitted!\n"
            f"  Expected actions from players: {sorted(self._expected)}\n"
            f"  Received actions from players: {sorted(self._received)}\n"
            f"  Missing actions from players:  {sorted(missing)}\n"
            f"\n"
            f"  This means _advance_selfplay called get_pending after\n"
            f"  submitting only one player's action. Both players must\n"
            f"  submit before Showdown resolves the turn."
        )
        self._turn_idx += 1
        if self._turn_idx >= len(self._script):
            self._finished = True
            return []
        turn = self._script[self._turn_idx]
        self._expected = {idx for idx, _ in turn}
        self._received = set()
        return list(turn)

    def get_result(self):
        return {
            "won": True, "turns": self._turn_idx + 1, "steps": self._step_count,
            "format": "gen1randombattle", "battle_tag": "mock-selfplay",
            "selfplay": True,
        }

    async def close(self):
        self._finished = True

    @property
    def is_finished(self):
        return self._finished


# ---- Tests: setup_state ----


class TestHooksSetup:
    """Test setup_state initializes state correctly for both modes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_creates_manager_and_battle(self):
        """play_mode=single: manager created, agent.battle is a Battle object."""
        mock_mgr = StrictMockHeuristicManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        assert state["manager"] is mock_mgr
        agent = state["_agents"][state["_current_agent_idx"]]
        assert isinstance(agent.battle, MockBattle), (
            f"Expected MockBattle, got {type(agent.battle).__name__}: {agent.battle!r}"
        )
        assert state["game_over"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_battle_is_not_tuple(self):
        """CRITICAL: agent.battle must be a Battle object, NOT a tuple.

        start_battle_selfplay() returns [(0, battle1), (1, battle2)].
        setup_state must unpack and store bare Battle objects in _AgentContext.

        Previously broken: s1, s2 = [(0,b1), (1,b2)] stored tuples.
        Fixed: _AgentContext.battle = pending[idx][1] extracts bare Battle object.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        agent = state["_agents"][state["_current_agent_idx"]]
        battle = agent.battle
        assert not isinstance(battle, tuple), (
            f"agent.battle is a tuple {battle!r}. "
            f"Expected a Battle object. setup_state is storing the raw "
            f"(player_idx, battle) tuple from start_battle_selfplay."
        )
        assert hasattr(battle, "available_moves"), (
            f"agent.battle has no available_moves. Type: {type(battle).__name__}"
        )
        assert hasattr(battle, "turn"), (
            f"agent.battle has no turn attribute. Type: {type(battle).__name__}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_game_over_when_battles_none(self):
        """When battles are None, game_over must be True.

        Previously broken: checked `if s1 is None` but s1 was (0, None) -- truthy.
        Fixed: uses `any(b is None for _, b in pending)`.
        """
        class FailManager:
            async def start_battle_selfplay(self, **kw):
                return [(0, None), (1, None)]
            async def close(self):
                pass

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=FailManager()):
            state = await env.setup_state({})

        assert state["game_over"] is True, (
            "When start_battle_selfplay returns None battles, game_over must be True. "
            "Current code checks `if s1 is None` but s1 is the tuple (0, None)."
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_current_agent_starts_zero(self):
        """Self-play must start with _current_agent_idx=0."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        assert state.get("_current_agent_idx") == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_battle_not_none(self):
        """Single mode setup must set a non-None battle when game starts."""
        mock_mgr = StrictMockHeuristicManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        agent = state["_agents"][state["_current_agent_idx"]]
        assert agent.battle is not None, "Battle should be set after setup"
        # Also verify it's not a tuple (defensive)
        assert not isinstance(agent.battle, (list, tuple))


# ---- Tests: single mode hooks cycle ----


class TestHooksSingleCycle:
    """Full hooks cycle for single mode: setup -> prompt -> step -> render."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_cycle_3_turns(self):
        """3-turn game. Verify trajectory length, rewards, fields."""
        mock_mgr = StrictMockHeuristicManager(game_turns=3)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        step_count = 0
        while not state["game_over"]:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            assert len(prompt) == 2, "Prompt should have system + user messages"
            assert prompt[0]["role"] == "system"
            assert prompt[1]["role"] == "user"
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step_count += 1
            assert step_count <= 50, "Runaway loop -- game should have ended"

        await env.render_completion(state)

        assert len(state["trajectory"]) == 3, (
            f"Expected 3 steps for 3-turn game, got {len(state['trajectory'])}"
        )
        assert state["reward"] == 1.0  # mock returns won=True

        for i, s in enumerate(state["trajectory"]):
            assert s["extras"]["agent_idx"] == 0, f"Step {i}: single mode = agent 0 only"
            assert s["reward"] == 1.0, f"Step {i}: win -> reward 1.0"
            assert "parsed_action" in s["extras"], f"Step {i}: missing extras.parsed_action"
            assert "force_switch" in s["extras"], f"Step {i}: missing extras.force_switch"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_over_stops_prompts(self):
        """After game ends, get_prompt_messages must return None."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        prompt = await env.get_prompt_messages(state)
        assert prompt is not None, "First prompt should not be None"
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "move tackle"}],
            "prompt": prompt, "tokens": {},
        })
        assert state["game_over"] is True, "1-turn game should be over"

        # After game_over, game_over() stop condition should fire
        assert await env.game_over(state) is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_on_garbage_input(self):
        """Garbage completion text -> fallback action, not crash."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "garbage no json"}],
            "prompt": [], "tokens": {},
        })

        assert state["trajectory"][0]["extras"]["parsed_action"] == "/choose move fallback", (
            f"Expected fallback, got: {state['trajectory'][0]['extras']['parsed_action']}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parsed_action_on_valid_input(self):
        """Valid move JSON -> parsed action recorded."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "thunderbolt"}'}],
            "prompt": [], "tokens": {},
        })

        assert state["trajectory"][0]["extras"]["parsed_action"] == "/choose move parsed_move"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_won_state_set_on_finish(self):
        """After game ends, state['won'] must be set from get_result."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "move tackle"}],
            "prompt": [], "tokens": {},
        })

        assert "won" in state, "state['won'] must be set after game over"
        assert state["won"] is True, "Mock manager returns won=True"


# ---- Tests: selfplay hooks cycle ----


class TestHooksSelfplayCycle:
    """Tests for the selfplay hooks path.

    These tests validate the fixed selfplay hooks flow:
    - setup_state unpacks tuples into _AgentContext objects correctly
    - _advance_selfplay buffers pending states
    - get_pending only called after all buffered actions submitted
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_cycle_both_players_act(self):
        """Both players must act each turn via hooks.

        Validates that the hooks path correctly processes both players
        per turn without crashing or violating the BattleManager contract.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        step_count = 0
        while not state["game_over"]:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step_count += 1
            assert step_count <= 100, "Possible deadlock -- 100 steps without end"

        await env.render_completion(state)

        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0) > 0, "P0 must have trajectory steps"
        assert len(p1) > 0, "P1 must have trajectory steps"
        assert p0[0]["reward"] != p1[0]["reward"], "P0 and P1 rewards must differ"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_advance_does_not_call_get_pending_prematurely(self):
        """_advance_selfplay must NOT call get_pending until both players acted.

        Directly tests _advance_selfplay with a manually correct state.
        The strict mock raises if the contract is violated.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        # Manually construct correct state using _AgentContext
        pending = await mock_mgr.start_battle_selfplay()
        agent0 = _AgentContext(agent_idx=0)
        agent0.battle = pending[0][1]
        agent1 = _AgentContext(agent_idx=1)
        agent1.battle = pending[1][1]

        state = {
            "trajectory": [],
            "game_over": False,
            "game_turn": 1,
            "won": None,
            "truncated": False,
            "manager": mock_mgr,
            "_current_agent_idx": 0,
            "_agents": [agent0, agent1],
            "_pending_states": pending,
        }

        action = MockAction("tackle")
        # This SHOULD buffer P2's state and NOT call get_pending
        await env._advance_selfplay(state, action, 0)

        assert state["_current_agent_idx"] == 1, (
            f"After P0 acts, _current_agent_idx should be 1, got {state.get('_current_agent_idx')}"
        )
        assert not state["game_over"], "Game should not be over after just P0's action"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_second_player_state_not_lost(self):
        """P2's buffered state must be used for the next prompt.

        After P0 acts, agent1.battle should be P2's battle from
        the buffered _pending_states, not from a new get_pending call.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        pending = await mock_mgr.start_battle_selfplay()
        p2_battle = pending[1][1]

        agent0 = _AgentContext(agent_idx=0)
        agent0.battle = pending[0][1]
        agent1 = _AgentContext(agent_idx=1)
        agent1.battle = pending[1][1]

        state = {
            "trajectory": [],
            "game_over": False,
            "game_turn": 1,
            "won": None,
            "truncated": False,
            "manager": mock_mgr,
            "_current_agent_idx": 0,
            "_agents": [agent0, agent1],
            "_pending_states": pending,
        }

        # P0 acts
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": [], "tokens": {},
        })

        # P2's battle should now be set from the buffered state
        current_agent = state["_agents"][state["_current_agent_idx"]]
        assert current_agent.battle is p2_battle, (
            f"P2's battle was lost. agent1.battle should be p2's MockBattle. "
            f"Got: {current_agent.battle!r}"
        )
        assert state["_current_agent_idx"] == 1

        # P2 should get a prompt
        prompt = await env.get_prompt_messages(state)
        assert prompt is not None, "P2 must get a prompt from buffered state"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_won_none_symmetric_rewards(self):
        """Self-play draw/crash (won=None): both players get reward_draw.

        Default reward_draw=0.0. With Phase 4, this uses configurable rewards.
        """
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        state = {
            "won": None, "truncated": False, "game_turn": 5,
            "trajectory": [
                {"extras": {"agent_idx": 0}},
                {"extras": {"agent_idx": 1}},
            ],
        }

        env._assign_rewards(state)

        p0_reward = state["trajectory"][0]["reward"]
        p1_reward = state["trajectory"][1]["reward"]
        assert p0_reward == p1_reward == 0.0, (
            f"For won=None, both players should get reward_draw (0.0). "
            f"Got P0={p0_reward}, P1={p1_reward}"
        )


# ---- Tests: strict mock validation ----


class TestStrictMockValidation:
    """Validate that the strict mocks themselves work correctly.

    These tests should ALWAYS PASS -- they verify the test infrastructure.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_catches_premature_get_pending(self):
        """Strict mock raises on premature get_pending."""
        mock = StrictMockSelfplayManager()
        await mock.start_battle_selfplay()

        # Submit only P1
        await mock.submit_selfplay_action(0, MockAction())

        with pytest.raises(AssertionError, match="before all actions"):
            await mock.get_pending_selfplay_states()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_allows_correct_usage(self):
        """Strict mock passes when contract is followed."""
        mock = StrictMockSelfplayManager()
        pending = await mock.start_battle_selfplay()
        assert len(pending) == 2

        for idx, _ in pending:
            await mock.submit_selfplay_action(idx, MockAction())

        next_pending = await mock.get_pending_selfplay_states()
        assert len(next_pending) == 2  # turn 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_game_ends_after_script(self):
        """Strict mock returns [] after all scripted turns."""
        mock = StrictMockSelfplayManager()  # 3 turns
        pending = await mock.start_battle_selfplay()

        for _ in range(3):  # play all 3 turns
            for idx, _ in pending:
                await mock.submit_selfplay_action(idx, MockAction())
            pending = await mock.get_pending_selfplay_states()

        assert pending == [], "Should be empty after all turns"
        assert mock.is_finished

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_rejects_duplicate_action(self):
        """Submitting for the same player twice raises."""
        mock = StrictMockSelfplayManager()
        await mock.start_battle_selfplay()

        await mock.submit_selfplay_action(0, MockAction())

        with pytest.raises(AssertionError, match="Duplicate"):
            await mock.submit_selfplay_action(0, MockAction())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_rejects_unexpected_player(self):
        """Submitting for a player not in expected set raises."""
        mock = StrictMockSelfplayManager(
            game_script=[[(0, MockBattle())]]  # only player 0
        )
        await mock.start_battle_selfplay()

        with pytest.raises(AssertionError, match="Unexpected"):
            await mock.submit_selfplay_action(1, MockAction())


# ---- Tests: standalone selfplay as reference ----


class TestSelfplayStandaloneContract:
    """Verify _run_selfplay_standalone follows the BattleManager contract.

    This is the REFERENCE implementation. The strict mock validates
    correctness. If this passes, we know the correct behavior is possible
    and the hooks path just needs to match it.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_follows_contract(self):
        """Standalone submits all actions before get_pending -- no assertion errors."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        assert result["won"] is True
        assert result["selfplay"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_both_players_get_steps(self):
        """Both players must have trajectory steps in standalone selfplay."""
        mock_mgr = StrictMockSelfplayManager()  # 3 turns
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        trajectory = []
        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), trajectory
        )

        p0 = [s for s in result["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in result["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0) == 3, f"P0 should have 3 steps (3 turns), got {len(p0)}"
        assert len(p1) == 3, f"P1 should have 3 steps (3 turns), got {len(p1)}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_opposite_rewards(self):
        """Standalone selfplay: P0 wins -> P0=1.0, P1=0.0."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        for s in result["trajectory"]:
            if s["extras"]["agent_idx"] == 0:
                assert s["reward"] == 1.0, f"P0 wins -> 1.0, got {s['reward']}"
            else:
                assert s["reward"] == 0.0, f"P1 loses -> 0.0, got {s['reward']}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_agent_indices_only_0_or_1(self):
        """All trajectory steps must have agent_idx 0 or 1."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        for i, s in enumerate(result["trajectory"]):
            assert s["extras"]["agent_idx"] in (0, 1), (
                f"Step {i}: agent_idx must be 0 or 1, got {s['extras']['agent_idx']}"
            )
