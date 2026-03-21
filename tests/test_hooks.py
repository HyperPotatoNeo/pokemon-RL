"""Tests for the MultiTurnEnv hooks path — the interface verifiers will use.

Tests setup_state → get_prompt_messages → add_trajectory_step →
render_completion for both heuristic and self-play modes, using mock
BattleManagers that ENFORCE the API contract.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    - Strict mocks ASSERT on contract violations (no silent deadlocks)
    - Tests verify values are the correct TYPE (not tuples where Battles expected)
    - Both positive and negative cases are checked
    - Every trajectory field is verified, not just length > 0

BUGS FIXED (these tests now pass):
    1. setup_state unpacks (idx, battle) tuples into bare Battle objects
    2. _advance_selfplay buffers pending states, calls get_pending only after all acted
    3. _pending_states consumed correctly — P2 state no longer lost
    4. selfplay won=None gives symmetric 0.0 rewards to both players
"""

import pytest
from unittest.mock import patch

from pokemon_rl.env import PokemonBattleEnv


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
    state['battle'] contains MockBattle instances, not (idx, MockBattle) tuples.
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
    """Mock StateTranslator that validates input types.

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
        if "move" in text.lower():
            return MockAction("parsed_move")
        return None

    def get_fallback_action(self, battle):
        return MockAction("fallback")


class StrictMockHeuristicManager:
    """Mock BattleManager for heuristic mode that enforces the API contract.

    Contract:
    1. start_battle() called once → returns Battle
    2. step(action) called per turn → returns (Battle, False) or (None, True)
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

    @property
    def is_finished(self):
        return self._finished


# ---- Tests: setup_state ----


class TestHooksSetup:
    """Test setup_state initializes state correctly for both modes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_battle_no_manager(self):
        """full_battle mode should NOT create a BattleManager."""
        env = PokemonBattleEnv(adapter=None, translator=None)
        state = await env.setup_state({})
        assert state["manager"] is None
        assert state["battle"] is None
        assert state["game_over"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_heuristic_creates_manager_and_battle(self):
        """turn_by_turn + heuristic: manager created, battle is a Battle object."""
        mock_mgr = StrictMockHeuristicManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        assert state["manager"] is mock_mgr
        assert isinstance(state["battle"], MockBattle), (
            f"Expected MockBattle, got {type(state['battle']).__name__}: {state['battle']!r}"
        )
        assert state["game_over"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_battle_is_not_tuple(self):
        """CRITICAL: state['battle'] must be a Battle object, NOT a tuple.

        start_battle_selfplay() returns [(0, battle1), (1, battle2)].
        setup_state must unpack and store bare Battle objects.

        Previously broken: s1, s2 = [(0,b1), (1,b2)] stored tuples.
        Fixed: pending[0][1] extracts bare Battle object.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        battle = state["battle"]
        assert not isinstance(battle, tuple), (
            f"state['battle'] is a tuple {battle!r}. "
            f"Expected a Battle object. setup_state is storing the raw "
            f"(player_idx, battle) tuple from start_battle_selfplay."
        )
        assert hasattr(battle, "available_moves"), (
            f"state['battle'] has no available_moves. Type: {type(battle).__name__}"
        )
        assert hasattr(battle, "turn"), (
            f"state['battle'] has no turn attribute. Type: {type(battle).__name__}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_game_over_when_battles_none(self):
        """When battles are None, game_over must be True.

        Previously broken: checked `if s1 is None` but s1 was (0, None) — truthy.
        Fixed: uses `any(b is None for _, b in pending)`.
        """
        class FailManager:
            async def start_battle_selfplay(self, **kw):
                return [(0, None), (1, None)]

        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=FailManager()):
            state = await env.setup_state({})

        assert state["game_over"] is True, (
            "When start_battle_selfplay returns None battles, game_over must be True. "
            "Current code checks `if s1 is None` but s1 is the tuple (0, None)."
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_current_player_starts_zero(self):
        """Self-play must start with current_player=0."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        assert state.get("current_player") == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_heuristic_battle_not_none(self):
        """Heuristic setup must set a non-None battle when game starts."""
        mock_mgr = StrictMockHeuristicManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        assert state["battle"] is not None, "Battle should be set after setup"
        # Also verify it's not a tuple (defensive)
        assert not isinstance(state["battle"], (list, tuple))


# ---- Tests: heuristic hooks cycle ----


class TestHooksHeuristicCycle:
    """Full hooks cycle for heuristic: setup → prompt → step → render."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_cycle_3_turns(self):
        """3-turn game. Verify trajectory length, rewards, fields."""
        mock_mgr = StrictMockHeuristicManager(game_turns=3)
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
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
            await env.add_trajectory_step(state, {"completion": '{"move": "tackle"}'})
            step_count += 1
            assert step_count <= 50, "Runaway loop — game should have ended"

        await env.render_completion(state)

        assert len(state["trajectory"]) == 3, (
            f"Expected 3 steps for 3-turn game, got {len(state['trajectory'])}"
        )
        assert state["decision_count"] == 3
        assert state["reward"] == 1.0  # mock returns won=True

        for i, s in enumerate(state["trajectory"]):
            assert s["player_idx"] == 0, f"Step {i}: heuristic = player 0 only"
            assert s["reward"] == 1.0, f"Step {i}: win → reward 1.0"
            assert "parsed_action" in s, f"Step {i}: missing parsed_action"
            assert "force_switch" in s, f"Step {i}: missing force_switch"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_over_stops_prompts(self):
        """After game ends, get_prompt_messages must return None."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        prompt = await env.get_prompt_messages(state)
        assert prompt is not None, "First prompt should not be None"
        await env.add_trajectory_step(state, {"completion": "move tackle"})
        assert state["game_over"] is True, "1-turn game should be over"

        prompt2 = await env.get_prompt_messages(state)
        assert prompt2 is None, "Prompt after game_over must be None"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_on_garbage_input(self):
        """Garbage completion text → fallback action, not crash."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {"completion": "garbage no json"})

        assert state["trajectory"][0]["parsed_action"] == "/choose move fallback", (
            f"Expected fallback, got: {state['trajectory'][0]['parsed_action']}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parsed_action_on_valid_input(self):
        """Valid move JSON → parsed action recorded."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {"completion": '{"move": "thunderbolt"}'})

        assert state["trajectory"][0]["parsed_action"] == "/choose move parsed_move"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_won_state_set_on_finish(self):
        """After game ends, state['won'] must be set from get_result."""
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="heuristic",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {"completion": "move tackle"})

        assert "won" in state, "state['won'] must be set after game over"
        assert state["won"] is True, "Mock manager returns won=True"


# ---- Tests: selfplay hooks cycle ----


class TestHooksSelfplayCycle:
    """Tests for the selfplay hooks path.

    These tests validate the fixed selfplay hooks flow:
    - setup_state unpacks tuples correctly
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
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({})

        step_count = 0
        while not state["game_over"]:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            await env.add_trajectory_step(state, {"completion": '{"move": "tackle"}'})
            step_count += 1
            assert step_count <= 100, "Possible deadlock — 100 steps without end"

        await env.render_completion(state)

        p1 = [s for s in state["trajectory"] if s["player_idx"] == 0]
        p2 = [s for s in state["trajectory"] if s["player_idx"] == 1]
        assert len(p1) > 0, "P1 must have trajectory steps"
        assert len(p2) > 0, "P2 must have trajectory steps"
        assert p1[0]["reward"] != p2[0]["reward"], "P1 and P2 rewards must differ"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_advance_does_not_call_get_pending_prematurely(self):
        """_advance_selfplay must NOT call get_pending until both players acted.

        Directly tests _advance_selfplay with a manually correct state.
        The strict mock raises if the contract is violated.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        # Manually construct correct state (bypassing setup_state tuple bug)
        pending = await mock_mgr.start_battle_selfplay()
        state = {
            "trajectory": [],
            "game_over": False,
            "turn": 1,
            "decision_count": 0,
            "won": None,
            "truncated": False,
            "parse_failure_count": 0,
            "manager": mock_mgr,
            "current_player": 0,
            "battle": pending[0][1],  # bare Battle, not tuple
            "_pending_states": pending,
        }

        action = MockAction("tackle")
        # This SHOULD buffer P2's state and NOT call get_pending
        await env._advance_selfplay(state, action, 0)

        assert state["current_player"] == 1, (
            f"After P1 acts, current_player should be 1, got {state.get('current_player')}"
        )
        assert not state["game_over"], "Game should not be over after just P1's action"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_second_player_state_not_lost(self):
        """P2's buffered state must be used for the next prompt.

        After P1 acts, state['battle'] should be P2's battle from
        the buffered _pending_states, not from a new get_pending call.
        """
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        pending = await mock_mgr.start_battle_selfplay()
        p2_battle = pending[1][1]

        state = {
            "trajectory": [],
            "game_over": False,
            "turn": 1,
            "decision_count": 0,
            "won": None,
            "truncated": False,
            "parse_failure_count": 0,
            "manager": mock_mgr,
            "current_player": 0,
            "battle": pending[0][1],
            "_pending_states": pending,
        }

        # P1 acts
        await env.add_trajectory_step(state, {"completion": '{"move": "tackle"}'})

        # P2's battle should now be set from the buffered state
        assert state["battle"] is p2_battle, (
            f"P2's battle was lost. state['battle'] should be p2's MockBattle. "
            f"Got: {state['battle']!r}"
        )
        assert state["current_player"] == 1

        # P2 should get a prompt
        prompt = await env.get_prompt_messages(state)
        assert prompt is not None, "P2 must get a prompt from buffered state"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_won_none_symmetric_rewards(self):
        """Self-play draw/crash (won=None): both players get reward_draw.

        Default reward_draw=0.0. With N1, this uses configurable rewards.
        """
        env = PokemonBattleEnv(
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        state = {
            "won": None, "truncated": False, "turn": 5, "decision_count": 10,
            "trajectory": [
                {"player_idx": 0},
                {"player_idx": 1},
            ],
        }

        await env.render_completion(state)

        p1_reward = state["trajectory"][0]["reward"]
        p2_reward = state["trajectory"][1]["reward"]
        assert p1_reward == p2_reward == 0.0, (
            f"For won=None, both players should get reward_draw (0.0). "
            f"Got P1={p1_reward}, P2={p2_reward}"
        )


# ---- Tests: strict mock validation ----


class TestStrictMockValidation:
    """Validate that the strict mocks themselves work correctly.

    These tests should ALWAYS PASS — they verify the test infrastructure.
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
        """Standalone submits all actions before get_pending — no assertion errors."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        assert result["won"] is True
        assert result["selfplay"] is True
        assert result["decision_count"] > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_both_players_get_steps(self):
        """Both players must have trajectory steps in standalone selfplay."""
        mock_mgr = StrictMockSelfplayManager()  # 3 turns
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        trajectory = []
        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), trajectory
        )

        p1 = [s for s in result["trajectory"] if s["player_idx"] == 0]
        p2 = [s for s in result["trajectory"] if s["player_idx"] == 1]
        assert len(p1) == 3, f"P1 should have 3 steps (3 turns), got {len(p1)}"
        assert len(p2) == 3, f"P2 should have 3 steps (3 turns), got {len(p2)}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_opposite_rewards(self):
        """Standalone selfplay: P1 wins → P1=1.0, P2=0.0."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        for s in result["trajectory"]:
            if s["player_idx"] == 0:
                assert s["reward"] == 1.0, f"P1 wins → 1.0, got {s['reward']}"
            else:
                assert s["reward"] == 0.0, f"P2 loses → 0.0, got {s['reward']}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standalone_player_indices_only_0_or_1(self):
        """All trajectory steps must have player_idx 0 or 1."""
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            translator=MockTranslator(),
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        result = await env._run_selfplay_standalone(
            mock_mgr, lambda b: MockAction(), []
        )

        for i, s in enumerate(result["trajectory"]):
            assert s["player_idx"] in (0, 1), (
                f"Step {i}: player_idx must be 0 or 1, got {s['player_idx']}"
            )
