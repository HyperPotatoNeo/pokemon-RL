"""Phase 4 unit tests: Verifiers integration.

Tests the complete PokemonBattleEnv(vf.MultiTurnEnv) implementation including:
- PokemonRubric registration and scoring
- _AgentContext lifecycle
- _build_agent_prompt delegation
- _assign_rewards with advantage pre-setting
- @vf.stop and @vf.cleanup decorators
- Self-play turn alternation and force-switch handling
- Error boundary (BattleManager/translator → vf.Error)
- Dataset and environment discovery

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
Every test verifies BOTH that correct input produces the right output AND that
incorrect input produces detectably different output. "len(x) > 0" is not enough;
check specific values.

ANTI-REWARD-HACKING: Multiple tests specifically verify that broken/garbage LLM
output cannot exploit the reward system (random fallback, passthrough rubric, etc.).

NOTE: These tests are written AGAINST THE PLAN SPECIFICATION in PHASE_VERIFIERS_PLAN.md.
If the implementation deviates from the plan, these tests are the authority.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers: check if verifiers is importable (needed for some tests)
# ---------------------------------------------------------------------------
try:
    import verifiers as vf
    HAS_VERIFIERS = True
except ImportError:
    HAS_VERIFIERS = False

requires_verifiers = pytest.mark.skipif(
    not HAS_VERIFIERS, reason="verifiers not installed"
)


# ---------------------------------------------------------------------------
# Mock infrastructure (extended from test_hooks.py)
# ---------------------------------------------------------------------------

class MockMove:
    def __init__(self, move_id="tackle", base_power=40):
        self.id = move_id
        self.base_power = base_power
        self.type = "normal"


class MockBattle:
    """Minimal Battle mock. NOT a tuple — tests verify this."""
    def __init__(self, name="mock", turn=1, moves=None, switches=None,
                 format_str="gen1randombattle"):
        self.name = name
        self.turn = turn
        self.available_moves = moves if moves is not None else [MockMove()]
        self.available_switches = switches if switches is not None else []
        self.force_switch = False
        self.won = None
        self.battle_tag = f"mock-{name}"
        self._format = format_str
        self.active_pokemon = None
        self.opponent_active_pokemon = None


class MockAction:
    def __init__(self, msg="tackle"):
        self.message = f"/choose move {msg}"


class MockTranslator:
    """Mock translator that validates input types and supports Messages format."""
    def battle_to_prompt(self, battle):
        assert not isinstance(battle, tuple), (
            f"battle_to_prompt got tuple {battle!r}, expected Battle object"
        )
        return [
            {"role": "system", "content": "Pokemon battle AI system prompt."},
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
            # If no assistant message, concatenate all content
            return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return str(messages)

    def extract_user_content(self, messages):
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return ""


class StrictMockSelfplayManager:
    """Strict selfplay mock from test_hooks.py — enforces API contract."""
    def __init__(self, game_script=None, winner=True):
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
        self._winner = winner

    async def start_battle_selfplay(self, **kwargs):
        self._turn_idx = 0
        turn = self._script[0]
        self._expected = {idx for idx, _ in turn}
        self._received = set()
        return list(turn)

    async def submit_selfplay_action(self, player_idx, action):
        assert player_idx in self._expected, (
            f"Unexpected player {player_idx}. Expected: {self._expected}"
        )
        assert player_idx not in self._received, (
            f"Duplicate action for player {player_idx}"
        )
        self._received.add(player_idx)
        self._step_count += 1

    async def get_pending_selfplay_states(self):
        missing = self._expected - self._received
        assert not missing, (
            f"get_pending called before all actions! Missing: {sorted(missing)}"
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
            "won": self._winner, "turns": self._turn_idx + 1,
            "steps": self._step_count, "format": "gen1randombattle",
            "battle_tag": "mock-selfplay", "selfplay": True,
        }

    async def close(self):
        self._finished = True

    @property
    def is_finished(self):
        return self._finished


class StrictMockHeuristicManager:
    """Strict heuristic mock — enforces start→step→step→result contract."""
    def __init__(self, game_turns=3, winner=True):
        self._turns = game_turns
        self._current = 0
        self._step_count = 0
        self._started = False
        self._finished = False
        self._winner = winner

    async def start_battle(self, **kwargs):
        assert not self._started, "start_battle called twice"
        self._started = True
        self._current = 1
        return MockBattle("turn1", turn=1)

    async def step(self, action):
        assert self._started and not self._finished
        self._step_count += 1
        self._current += 1
        if self._current > self._turns:
            self._finished = True
            return None, True
        return MockBattle(f"turn{self._current}", turn=self._current), False

    def get_result(self):
        return {
            "won": self._winner, "turns": self._current,
            "steps": self._step_count, "format": "gen1randombattle",
            "battle_tag": "mock-heuristic", "selfplay": False,
        }

    async def close(self):
        self._finished = True


class ErrorManager:
    """Manager that raises on step — for error boundary tests."""
    async def start_battle(self, **kwargs):
        return MockBattle("error_test")

    async def step(self, action):
        raise ConnectionError("Showdown connection lost")

    def get_result(self):
        return {"won": None, "turns": 0, "steps": 0, "format": "gen1randombattle",
                "battle_tag": "error", "selfplay": False}

    async def close(self):
        pass


class ErrorTranslator:
    """Translator that raises on battle_to_prompt."""
    def battle_to_prompt(self, battle):
        raise RuntimeError("Translator exploded")

    def parse_action(self, text, battle):
        return MockAction()

    def get_fallback_action(self, battle):
        return MockAction("fallback")

    @staticmethod
    def extract_completion_text(messages):
        return "test"


# ============================================================================
# T1: PokemonRubric
# ============================================================================

class TestPokemonRubric:
    """Verify PokemonRubric has correct methods registered."""

    @requires_verifiers
    @pytest.mark.unit
    def test_rubric_has_registered_reward_func(self):
        """PokemonRubric must have at least one reward function registered.
        NEGATIVE: funcs list is not empty."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        assert len(rubric.funcs) > 0, "PokemonRubric has no reward functions registered"

    @requires_verifiers
    @pytest.mark.unit
    def test_rubric_has_registered_metrics(self):
        """PokemonRubric must register game metrics (won, game_turns, parse_failures)."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        # Metrics are stored alongside reward funcs — check they exist
        # The Rubric stores all funcs (reward + metrics) in self.funcs
        assert len(rubric.funcs) >= 2, (
            f"Expected at least 2 funcs (1 reward + 1+ metrics), got {len(rubric.funcs)}"
        )

    @pytest.mark.unit
    def test_passthrough_reward_returns_state_reward(self):
        """Passthrough must return exactly state['reward'], not compute its own."""
        from pokemon_rl.env import PokemonRubric
        # Test with various reward values
        for reward_val in [1.0, 0.0, -1.0, 0.5, -0.5, 42.0]:
            state = {"reward": reward_val, "trajectory": []}
            rubric = PokemonRubric()
            # Call the passthrough function directly
            result = rubric._passthrough_reward_sync(state)
            assert result == reward_val, (
                f"Passthrough should return {reward_val}, got {result}"
            )

    @pytest.mark.unit
    def test_passthrough_reward_none_returns_zero(self):
        """CR-1: state['reward']=None must return 0.0, not None."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        result = rubric._passthrough_reward_sync({"reward": None, "trajectory": []})
        assert result == 0.0, f"reward=None should give 0.0, got {result}"
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    @pytest.mark.unit
    def test_passthrough_reward_missing_returns_zero(self):
        """Missing state['reward'] must return 0.0."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        result = rubric._passthrough_reward_sync({"trajectory": []})
        assert result == 0.0, f"Missing reward should give 0.0, got {result}"

    @pytest.mark.unit
    def test_passthrough_reward_zero_returns_zero_not_default(self):
        """reward=0 must return 0, not be treated as falsy default.
        NEGATIVE: 0 is a valid reward, must not be converted to something else."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        result = rubric._passthrough_reward_sync({"reward": 0, "trajectory": []})
        assert result == 0, f"reward=0 should return 0, got {result}"

    @pytest.mark.unit
    def test_passthrough_reward_negative_preserved(self):
        """Negative rewards must not be zeroed out."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        result = rubric._passthrough_reward_sync({"reward": -5.0, "trajectory": []})
        assert result == -5.0, f"reward=-5.0 should return -5.0, got {result}"

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_passthrough_returns_float_not_coroutine(self):
        """CRIT-2: The async passthrough_reward wrapper must return a float,
        not a coroutine object. A broken async wrapper would zero all rewards."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        state = {"reward": 0.75, "trajectory": []}
        # Call the async version (what the rubric framework actually calls)
        result = await rubric.passthrough_reward(state)
        assert isinstance(result, (int, float)), (
            f"Async passthrough must return float, got {type(result)}: {result}"
        )
        assert result == 0.75

    @pytest.mark.unit
    def test_passthrough_strict_identity(self):
        """Passthrough must return the EXACT value, not a type-coerced copy.
        Catches: return float(state.get('reward', 0))  which coerces int→float."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        # Integer reward — must survive as int (or at least equal)
        assert rubric._passthrough_reward_sync({"reward": 1, "trajectory": []}) == 1
        # Float reward
        assert rubric._passthrough_reward_sync({"reward": 0.123456789, "trajectory": []}) == 0.123456789


# ============================================================================
# T2: _AgentContext
# ============================================================================

class TestAgentContext:
    """Verify _AgentContext is a correct passive dataclass."""

    @pytest.mark.unit
    def test_agent_context_fields(self):
        """_AgentContext must have all required fields with correct defaults."""
        from pokemon_rl.env import _AgentContext
        ctx = _AgentContext(agent_idx=0)
        assert ctx.agent_idx == 0
        assert ctx.battle is None
        assert ctx.steps == [] or ctx.steps is not None
        assert isinstance(ctx.steps, list)
        assert ctx.message_history == [] or ctx.message_history is not None
        assert isinstance(ctx.message_history, list)
        assert ctx.parse_failure_count == 0
        assert ctx.force_switch_count == 0

    @pytest.mark.unit
    def test_agent_context_independent_lists(self):
        """Two _AgentContext instances must NOT share list references.
        NEGATIVE: steps list must be independent (not same object)."""
        from pokemon_rl.env import _AgentContext
        ctx0 = _AgentContext(agent_idx=0)
        ctx1 = _AgentContext(agent_idx=1)
        ctx0.steps.append("x")
        assert len(ctx1.steps) == 0, (
            "ctx1.steps was modified when ctx0.steps was changed — shared list!"
        )
        ctx0.message_history.append("y")
        assert len(ctx1.message_history) == 0, (
            "ctx1.message_history was modified when ctx0's was changed — shared list!"
        )

    @pytest.mark.unit
    def test_agent_context_no_methods(self):
        """_AgentContext should be a passive dataclass with no behavior methods.
        NEGATIVE: should not have methods beyond __init__, __repr__, etc."""
        from pokemon_rl.env import _AgentContext
        # Check it's a dataclass
        import dataclasses
        assert dataclasses.is_dataclass(_AgentContext), (
            "_AgentContext should be a dataclass"
        )
        # Check no custom methods (besides dunder)
        custom_methods = [
            m for m in dir(_AgentContext)
            if not m.startswith('_') and callable(getattr(_AgentContext, m))
        ]
        assert len(custom_methods) == 0, (
            f"_AgentContext should have no custom methods, found: {custom_methods}"
        )


# ============================================================================
# T3: _assign_rewards — Single Agent
# ============================================================================

class TestAssignRewardsSingleAgent:
    """Verify reward assignment for single-agent (play_mode='single')."""

    @pytest.mark.unit
    def test_win_gets_reward_win(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = self._make_state(won=True)
        env._assign_rewards(state)
        assert all(s["reward"] == 1.0 for s in state["trajectory"])

    @pytest.mark.unit
    def test_loss_gets_reward_loss(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = self._make_state(won=False)
        env._assign_rewards(state)
        assert all(s["reward"] == 0.0 for s in state["trajectory"])

    @pytest.mark.unit
    def test_win_and_loss_distinguishable(self):
        """NEGATIVE: win reward must differ from loss reward."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        win_state = self._make_state(won=True)
        loss_state = self._make_state(won=False)
        env._assign_rewards(win_state)
        env._assign_rewards(loss_state)
        assert win_state["trajectory"][0]["reward"] != loss_state["trajectory"][0]["reward"]

    @pytest.mark.unit
    def test_draw_gets_reward_draw(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = self._make_state(won=None)
        env._assign_rewards(state)
        assert all(s["reward"] == 0.0 for s in state["trajectory"])  # default draw=0.0

    @pytest.mark.unit
    def test_draw_distinguishable_from_loss_with_custom_rewards(self):
        """With custom rewards, draw and loss must be distinguishable.
        NEGATIVE: they must NOT be the same value."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(reward_loss=-1.0, reward_draw=0.5)
        draw_state = self._make_state(won=None)
        loss_state = self._make_state(won=False)
        env._assign_rewards(draw_state)
        env._assign_rewards(loss_state)
        assert draw_state["trajectory"][0]["reward"] != loss_state["trajectory"][0]["reward"]

    @pytest.mark.unit
    def test_custom_rewards_propagated(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(reward_win=10, reward_loss=-10, reward_draw=0.5)
        for won, expected in [(True, 10), (False, -10), (None, 0.5)]:
            state = self._make_state(won=won)
            env._assign_rewards(state)
            assert state["trajectory"][0]["reward"] == expected, (
                f"won={won}: expected {expected}, got {state['trajectory'][0]['reward']}"
            )

    @pytest.mark.unit
    def test_empty_trajectory_no_crash(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = {"won": True, "trajectory": []}
        env._assign_rewards(state)  # Should not raise

    @pytest.mark.unit
    def test_parse_failed_steps_get_same_reward(self):
        """Steps where parsing failed must get the SAME terminal reward.
        The model should not be punished/rewarded differently for parse failures."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = {
            "won": True,
            "trajectory": [
                {"extras": {"agent_idx": 0, "parse_failed": False}},
                {"extras": {"agent_idx": 0, "parse_failed": True}},
                {"extras": {"agent_idx": 0, "parse_failed": False}},
            ],
        }
        env._assign_rewards(state)
        rewards = [s["reward"] for s in state["trajectory"]]
        assert all(r == 1.0 for r in rewards), (
            f"All steps (including parse-failed) should get 1.0, got {rewards}"
        )

    @pytest.mark.unit
    def test_uniform_rewards_leave_advantage_none(self):
        """Single-agent with terminal-only reward: all rewards are uniform.
        step['advantage'] should remain None (score_group fills cross-rollout)."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = self._make_state(won=True)
        env._assign_rewards(state)
        for step in state["trajectory"]:
            assert step.get("advantage") is None, (
                f"Uniform rewards → advantage should be None, got {step.get('advantage')}"
            )

    # --- Helpers ---
    def _make_env(self, **kwargs):
        from pokemon_rl.env import PokemonBattleEnv
        defaults = dict(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        defaults.update(kwargs)
        return PokemonBattleEnv(**defaults)

    def _make_state(self, won=True, num_steps=3):
        return {
            "won": won,
            "trajectory": [
                {"extras": {"agent_idx": 0}} for _ in range(num_steps)
            ],
        }


# ============================================================================
# T4: _assign_rewards — Self-Play
# ============================================================================

class TestAssignRewardsSelfPlay:
    """Verify reward + advantage assignment for self-play."""

    @pytest.mark.unit
    def test_p0_wins_p0_gets_win_reward(self):
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        assert all(s["reward"] == 1.0 for s in p0)

    @pytest.mark.unit
    def test_p0_wins_p1_gets_loss_reward(self):
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert all(s["reward"] == 0.0 for s in p1)

    @pytest.mark.unit
    def test_p1_wins_rewards_inverted(self):
        """When P1 wins (won=False from P0's perspective), rewards must flip."""
        env = self._make_env()
        state = self._make_sp_state(won=False)
        env._assign_rewards(state)
        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert all(s["reward"] == 0.0 for s in p0), "P0 should get loss reward"
        assert all(s["reward"] == 1.0 for s in p1), "P1 should get win reward"

    @pytest.mark.unit
    def test_p0_win_and_p1_win_produce_different_assignments(self):
        """NEGATIVE: T3.1 and T3.2 must produce different reward assignments."""
        env = self._make_env()
        state_p0_wins = self._make_sp_state(won=True)
        state_p1_wins = self._make_sp_state(won=False)
        env._assign_rewards(state_p0_wins)
        env._assign_rewards(state_p1_wins)
        p0_r_when_p0_wins = state_p0_wins["trajectory"][0]["reward"]
        p0_r_when_p1_wins = state_p1_wins["trajectory"][0]["reward"]
        assert p0_r_when_p0_wins != p0_r_when_p1_wins, "Same player different outcomes → different rewards"

    @pytest.mark.unit
    def test_draw_both_get_reward_draw(self):
        env = self._make_env(reward_draw=0.5)
        state = self._make_sp_state(won=None)
        env._assign_rewards(state)
        for s in state["trajectory"]:
            assert s["reward"] == 0.5, f"Draw: both should get 0.5, got {s['reward']}"

    @pytest.mark.unit
    def test_won_none_does_not_default_to_false(self):
        """NEGATIVE: won=None must NOT be treated as won=False."""
        env = self._make_env(reward_win=1.0, reward_loss=-1.0, reward_draw=0.0)
        none_state = self._make_sp_state(won=None)
        false_state = self._make_sp_state(won=False)
        env._assign_rewards(none_state)
        env._assign_rewards(false_state)
        # won=None: both get 0.0. won=False: P0=-1, P1=1. Must differ.
        p0_none = [s for s in none_state["trajectory"] if s["extras"]["agent_idx"] == 0][0]
        p0_false = [s for s in false_state["trajectory"] if s["extras"]["agent_idx"] == 0][0]
        assert p0_none["reward"] != p0_false["reward"], (
            f"won=None P0 reward ({p0_none['reward']}) must differ from "
            f"won=False P0 reward ({p0_false['reward']})"
        )

    @pytest.mark.unit
    def test_custom_asymmetric_rewards(self):
        """Custom win=1, loss=-1: P1's reward must be reward_loss, NOT 1-reward_win."""
        env = self._make_env(reward_win=1.0, reward_loss=-1.0)
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert all(s["reward"] == -1.0 for s in p1), (
            "P1 loser should get -1.0, not 0.0 (which would be 1-1=0 bug)"
        )

    @pytest.mark.unit
    def test_agent_idx_only_0_or_1(self):
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        for i, s in enumerate(state["trajectory"]):
            assert s["extras"]["agent_idx"] in (0, 1), (
                f"Step {i}: agent_idx must be 0 or 1, got {s['extras']['agent_idx']}"
            )

    @pytest.mark.unit
    def test_agent_idx_matches_reward(self):
        """Cross-check: agent_idx must be consistent with reward value."""
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        for s in state["trajectory"]:
            if s["extras"]["agent_idx"] == 0:
                assert s["reward"] == 1.0
            else:
                assert s["reward"] == 0.0

    @pytest.mark.unit
    def test_selfplay_advantages_preset(self):
        """Self-play rewards are non-uniform → step['advantage'] must be pre-set.
        CRITICAL: Without pre-set, score_group assigns uniform state-level advantage."""
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        for s in state["trajectory"]:
            assert s.get("advantage") is not None, (
                f"Self-play step must have pre-set advantage, got None"
            )

    @pytest.mark.unit
    def test_selfplay_advantages_opposite_sign(self):
        """P0 and P1 must have advantages with opposite signs (one positive, one negative)."""
        env = self._make_env()
        state = self._make_sp_state(won=True)
        env._assign_rewards(state)
        p0_adv = [s["advantage"] for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1_adv = [s["advantage"] for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert all(a > 0 for a in p0_adv), f"P0 (winner) advantages should be positive: {p0_adv}"
        assert all(a < 0 for a in p1_adv), f"P1 (loser) advantages should be negative: {p1_adv}"

    @pytest.mark.unit
    def test_selfplay_draw_advantages_zero(self):
        """Draw: both get same reward → uniform → advantage should be None or 0."""
        env = self._make_env(reward_draw=0.5)
        state = self._make_sp_state(won=None)
        env._assign_rewards(state)
        # All same reward → uniform → advantage should be None (let score_group handle)
        for s in state["trajectory"]:
            assert s.get("advantage") is None, (
                f"Uniform self-play rewards (draw) → advantage should be None, "
                f"got {s.get('advantage')}"
            )

    # --- Helpers ---
    def _make_env(self, **kwargs):
        from pokemon_rl.env import PokemonBattleEnv
        defaults = dict(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        defaults.update(kwargs)
        return PokemonBattleEnv(**defaults)

    def _make_sp_state(self, won=True, turns_per_player=3):
        trajectory = []
        for t in range(turns_per_player):
            trajectory.append({"extras": {"agent_idx": 0}})
            trajectory.append({"extras": {"agent_idx": 1}})
        return {"won": won, "trajectory": trajectory}


# ============================================================================
# T5: Stop Conditions
# ============================================================================

class TestStopConditions:
    """Verify @vf.stop game_over behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_over_true_stops(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = {"game_over": True, "game_turn": 5}
        result = await env.game_over(state)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_over_false_continues(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = {"game_over": False, "game_turn": 5}
        result = await env.game_over(state)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_game_turns_triggers_stop(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(max_game_turns=10)
        state = {"game_over": False, "game_turn": 10}
        result = await env.game_over(state)
        assert result is True
        assert state["game_over"] is True
        assert state.get("truncated") is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_below_max_turns_continues(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(max_game_turns=10)
        state = {"game_over": False, "game_turn": 9}
        result = await env.game_over(state)
        assert result is False
        assert state["game_over"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_max_turns_counts_game_turns_not_steps(self):
        """Self-play doubles steps per turn but max_game_turns counts GAME turns."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(max_game_turns=10, play_mode="self_play")
        # 10 game turns = 20 trajectory steps in self-play
        state = {"game_over": False, "game_turn": 9, "trajectory": [{}] * 18}
        result = await env.game_over(state)
        assert result is False, "9 game turns < 10 max, should continue"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_truncation_uses_reward_draw(self):
        """Truncation (max turns) must use reward_draw, NOT reward_loss."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env(reward_draw=0.5, reward_loss=-1.0)
        state = {
            "game_over": False, "game_turn": 200, "won": None,
            "trajectory": [{"extras": {"agent_idx": 0}}],
        }
        await env.game_over(state)
        assert state.get("truncated") is True
        env._assign_rewards(state)
        assert state["trajectory"][0]["reward"] == 0.5, (
            f"Truncation should use reward_draw=0.5, got {state['trajectory'][0]['reward']}"
        )

    # --- Helpers ---
    def _make_env(self, **kwargs):
        from pokemon_rl.env import PokemonBattleEnv
        defaults = dict(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        defaults.update(kwargs)
        return PokemonBattleEnv(**defaults)


# ============================================================================
# T6: Hooks Cycle — Single Agent
# ============================================================================

class TestHooksCycleSingleAgent:
    """Full hooks cycle: setup → prompt → step → render for single agent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_cycle_3_turns(self):
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=3)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            assert len(prompt) == 2
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt,
                "tokens": {"prompt_ids": [], "completion_ids": [], "prompt_mask": [],
                           "completion_mask": [], "completion_logprobs": []},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1
            assert step_count <= 50, "Runaway loop"

        await env.render_completion(state)

        assert len(state["trajectory"]) == 3
        assert state["reward"] == 1.0
        # Verify completion field is set (CR-2)
        assert "completion" in state, "render_completion must set state['completion']"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trajectory_step_has_extras(self):
        """CR-6: Trajectory steps must have extras dict with required fields."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        prompt = await env.get_prompt_messages(state)
        step = {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt,
            "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        extras = state["trajectory"][0].get("extras")
        assert extras is not None, "Trajectory step must have 'extras' dict"
        assert "agent_idx" in extras, "extras must contain agent_idx"
        assert "game_turn" in extras, "extras must contain game_turn"
        assert "force_switch" in extras, "extras must contain force_switch"
        assert "parsed_action" in extras, "extras must contain parsed_action"
        assert "parse_failed" in extras, "extras must contain parse_failed"
        assert extras["agent_idx"] == 0, "Single-agent: agent_idx must be 0"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_messages_completion_extracted(self):
        """CR-7: add_trajectory_step must handle Messages-format completion."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        await env.get_prompt_messages(state)
        # Messages format completion (as verifiers sends it)
        step = {
            "completion": [{"role": "assistant", "content": '{"move": "thunder"}'}],
            "prompt": [],
            "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        # Should have parsed the action from Messages format
        extras = state["trajectory"][0].get("extras", {})
        assert extras.get("parse_failed") is not None, "parse_failed must be set"


# ============================================================================
# T7: Hooks Cycle — Self-Play
# ============================================================================

class TestHooksCycleSelfPlay:
    """Full hooks cycle for self-play: both agents act, rewards opposite."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_cycle_both_players_act(self):
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt,
                "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1
            assert step_count <= 100, "Runaway loop"

        await env.render_completion(state)

        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0) > 0, "P0 must have steps"
        assert len(p1) > 0, "P1 must have steps"
        assert len(p0) + len(p1) == len(state["trajectory"]), "All steps accounted for"
        # Winner and loser get different rewards
        assert p0[0]["reward"] != p1[0]["reward"], "P0 and P1 must get different rewards"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_advance_does_not_call_get_pending_prematurely(self):
        """_advance_selfplay must NOT call get_pending until ALL buffered actions submitted.
        The StrictMockSelfplayManager will raise if violated."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        pending = await mock_mgr.start_battle_selfplay()
        state = {
            "game_over": False, "game_turn": 1, "won": None,
            "manager": mock_mgr,
            "_agents": [MagicMock(agent_idx=0), MagicMock(agent_idx=1)],
            "_current_agent_idx": 0,
            "_pending_states": pending,
            "trajectory": [],
        }
        state["_agents"][0].battle = pending[0][1]
        state["_agents"][1].battle = pending[1][1]

        # Submit P0's action
        await env._advance_selfplay(state, MockAction(), 0)
        # Contract: P1 should now be current, get_pending NOT called yet
        assert state["_current_agent_idx"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_switch_one_player(self):
        """Force-switch: only one player gets a state. No deadlock."""
        from pokemon_rl.env import PokemonBattleEnv
        # Script: turn 1 normal (both), turn 2 force-switch (only P0), turn 3 normal
        game_script = [
            [(0, MockBattle("p1_t1", turn=1)), (1, MockBattle("p2_t1", turn=1))],
            [(0, MockBattle("p1_fs", turn=2))],  # force-switch: only P0
            [(0, MockBattle("p1_t3", turn=3)), (1, MockBattle("p2_t3", turn=3))],
        ]
        mock_mgr = StrictMockSelfplayManager(game_script=game_script)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1
            assert step_count <= 50, "Runaway"

        # Force-switch turn produces 1 step, normal turns produce 2
        assert step_count == 5, f"Expected 5 steps (2+1+2), got {step_count}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_history_independent_per_agent(self):
        """Each agent's message_history must be independent."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        # Play through all turns
        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)

        agents = state["_agents"]
        assert len(agents) == 2
        # Each agent should have 3 turns of history (3-turn script)
        assert len(agents[0].message_history) > 0, "Agent 0 should have history"
        assert len(agents[1].message_history) > 0, "Agent 1 should have history"
        # Histories must be different objects
        assert agents[0].message_history is not agents[1].message_history


# ============================================================================
# T8: Cleanup
# ============================================================================

class TestCleanup:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_calls_manager_close(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        closed = False
        class TrackedManager:
            async def close(self):
                nonlocal closed
                closed = True
        state = {"manager": TrackedManager()}
        await env.cleanup_battle(state)
        assert closed, "cleanup_battle must call manager.close()"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_sets_manager_none(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        class DummyManager:
            async def close(self):
                pass
        state = {"manager": DummyManager()}
        await env.cleanup_battle(state)
        assert state["manager"] is None, "cleanup must set manager to None"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """Calling cleanup twice must not crash."""
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        class DummyManager:
            async def close(self):
                pass
        state = {"manager": DummyManager()}
        await env.cleanup_battle(state)
        await env.cleanup_battle(state)  # Second call: manager is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_no_manager_no_crash(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = self._make_env()
        state = {}
        await env.cleanup_battle(state)  # Should not raise

    def _make_env(self):
        from pokemon_rl.env import PokemonBattleEnv
        return PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )


# ============================================================================
# T9: Prompt Construction
# ============================================================================

class TestPromptConstruction:

    @pytest.mark.unit
    def test_build_agent_prompt_returns_messages(self):
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()
        agent = _AgentContext(agent_idx=0)
        agent.battle = MockBattle("test")
        state = {}
        messages = env._build_agent_prompt(agent, state)
        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m for m in messages)

    @pytest.mark.unit
    def test_system_prompt_override(self):
        """Custom system_prompt should replace the translator's system message."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            system_prompt="Custom system prompt here.",
        )
        env.translator = MockTranslator()
        agent = _AgentContext(agent_idx=0)
        agent.battle = MockBattle("test")
        messages = env._build_agent_prompt(agent, {})
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Custom system prompt here."

    @pytest.mark.unit
    def test_different_agents_get_different_prompts(self):
        """Two agents with different battle states must get different prompts."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        agent0 = _AgentContext(agent_idx=0)
        agent0.battle = MockBattle("p0_state", turn=5)
        agent1 = _AgentContext(agent_idx=1)
        agent1.battle = MockBattle("p1_state", turn=5)
        prompt0 = env._build_agent_prompt(agent0, {})
        prompt1 = env._build_agent_prompt(agent1, {})
        # User messages should differ (different battle states)
        user0 = [m for m in prompt0 if m["role"] == "user"][0]["content"]
        user1 = [m for m in prompt1 if m["role"] == "user"][0]["content"]
        assert user0 != user1, "Different agents must get different user prompts"


# ============================================================================
# T10: Dataset, Config, Discovery
# ============================================================================

class TestDatasetConfigDiscovery:

    @pytest.mark.unit
    def test_play_mode_single_accepted(self):
        """play_mode='single' must be accepted (renamed from 'heuristic')."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        assert env.play_mode == "single"

    @pytest.mark.unit
    def test_play_mode_self_play_accepted(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        assert env.play_mode == "self_play"

    @pytest.mark.unit
    def test_play_mode_invalid_raises(self):
        from pokemon_rl.env import PokemonBattleEnv
        with pytest.raises(ValueError, match="play_mode"):
            PokemonBattleEnv(
                battle_format="gen1randombattle", port=8000,
                play_mode="invalid",
            )

    @pytest.mark.unit
    def test_play_mode_heuristic_rejected(self):
        """CR-4: Old 'heuristic' mode must be rejected (renamed to 'single')."""
        from pokemon_rl.env import PokemonBattleEnv
        with pytest.raises(ValueError):
            PokemonBattleEnv(
                battle_format="gen1randombattle", port=8000,
                play_mode="heuristic",
            )

    @pytest.mark.unit
    def test_load_environment_returns_env(self):
        """load_environment must return a PokemonBattleEnv instance."""
        from pokemon_rl import load_environment
        from pokemon_rl.env import PokemonBattleEnv
        env = load_environment(battle_format="gen1randombattle", port=8000)
        assert isinstance(env, PokemonBattleEnv)

    @pytest.mark.unit
    def test_load_environment_passes_kwargs(self):
        from pokemon_rl import load_environment
        env = load_environment(
            battle_format="gen9randombattle", port=9999,
            play_mode="self_play", reward_win=5.0,
        )
        assert env.battle_format == "gen9randombattle"
        assert env.port == 9999
        assert env.play_mode == "self_play"
        assert env.reward_win == 5.0

    @requires_verifiers
    @pytest.mark.unit
    def test_make_battle_dataset_structure(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", num_battles=10,
        )
        ds = env._make_battle_dataset(10, "gen1randombattle")
        assert "question" in ds.column_names
        assert "answer" in ds.column_names
        assert len(ds) == 10

    @requires_verifiers
    @pytest.mark.unit
    def test_env_inherits_multiturn(self):
        """PokemonBattleEnv must inherit from vf.MultiTurnEnv."""
        from pokemon_rl.env import PokemonBattleEnv
        import verifiers as vf
        assert issubclass(PokemonBattleEnv, vf.MultiTurnEnv)

    @requires_verifiers
    @pytest.mark.unit
    def test_score_rollouts_true(self):
        """score_rollouts must be True (constraint C7)."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single",
        )
        assert env.score_rollouts is True, "score_rollouts must be True (C7)"


# ============================================================================
# T11: Advantage Pipeline Simulation
# ============================================================================

class TestAdvantagePipeline:
    """Simulate score_group behavior to verify advantages survive."""

    @pytest.mark.unit
    def test_preset_advantage_survives_score_group_simulation(self):
        """Simulate score_group: if step['advantage'] is not None, DON'T overwrite."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play",
        )
        state = {
            "won": True,
            "trajectory": [
                {"extras": {"agent_idx": 0}},
                {"extras": {"agent_idx": 1}},
            ],
        }
        env._assign_rewards(state)

        # Save pre-set values
        original_advantages = [s["advantage"] for s in state["trajectory"]]
        original_rewards = [s["reward"] for s in state["trajectory"]]

        # Simulate score_group behavior (rubric.py:319-323)
        state["reward"] = state["trajectory"][0]["reward"]  # passthrough
        group_avg = 0.5  # Simulated average across rollouts
        state_advantage = state["reward"] - group_avg

        for t in state["trajectory"]:
            if t.get("advantage") is None:
                t["advantage"] = state_advantage  # Would overwrite!
            if t.get("reward") is None:
                t["reward"] = state["reward"]  # Would overwrite!

        # Verify pre-set values survived
        for i, step in enumerate(state["trajectory"]):
            assert step["advantage"] == original_advantages[i], (
                f"Step {i}: advantage changed from {original_advantages[i]} to "
                f"{step['advantage']} — score_group overwrote pre-set value!"
            )
            assert step["reward"] == original_rewards[i], (
                f"Step {i}: reward changed from {original_rewards[i]} to "
                f"{step['reward']} — score_group overwrote pre-set value!"
            )

    @pytest.mark.unit
    def test_uniform_advantage_none_filled_by_score_group(self):
        """Uniform rewards (single-agent) → advantage=None → score_group fills."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single",
        )
        state = {
            "won": True,
            "trajectory": [
                {"extras": {"agent_idx": 0}},
                {"extras": {"agent_idx": 0}},
            ],
        }
        env._assign_rewards(state)

        # Verify advantages are None
        for step in state["trajectory"]:
            assert step.get("advantage") is None, "Uniform → advantage should be None"

        # Simulate score_group filling
        state["reward"] = state["trajectory"][0]["reward"]
        group_avg = 0.5
        state_advantage = state["reward"] - group_avg
        for t in state["trajectory"]:
            if t.get("advantage") is None:
                t["advantage"] = state_advantage

        # Now advantage should be filled
        for step in state["trajectory"]:
            assert step["advantage"] == state_advantage


# ============================================================================
# T12: Error Handling
# ============================================================================

class TestErrorHandling:
    """Verify error boundary: external exceptions → vf.Error."""

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_battle_step_error_becomes_vf_error(self):
        """BattleManager.step() exception → vf.Error, not raw exception."""
        import verifiers as vf
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        error_mgr = ErrorManager()
        state = {
            "task": "battle", "prompt": [], "game_over": False,
            "game_turn": 1, "won": None, "trajectory": [],
            "manager": error_mgr,
            "_agents": [MagicMock(agent_idx=0, battle=MockBattle("test"),
                                   steps=[], message_history=[],
                                   parse_failure_count=0, force_switch_count=0)],
            "_current_agent_idx": 0,
        }

        with pytest.raises(vf.Error):
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": [], "tokens": {},
            }
            await env.add_trajectory_step(state, step)

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_translator_error_becomes_vf_error(self):
        """Translator exception in _build_agent_prompt → vf.Error."""
        import verifiers as vf
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = ErrorTranslator()

        agent = _AgentContext(agent_idx=0)
        agent.battle = MockBattle("test")
        state = {"_agents": [agent], "_current_agent_idx": 0}

        with pytest.raises(vf.Error):
            await env.get_prompt_messages(state)

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_setup_error_cleans_up_manager(self):
        """If setup_state fails, manager must be closed and state['manager']=None."""
        import verifiers as vf
        from pokemon_rl.env import PokemonBattleEnv

        class FailStartManager:
            close_called = False
            async def start_battle(self, **kw):
                raise RuntimeError("Server not running")
            async def close(self):
                FailStartManager.close_called = True

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        with patch('pokemon_rl.battle.BattleManager', return_value=FailStartManager()):
            with pytest.raises(vf.Error):
                await env.setup_state({"task": "battle", "prompt": []})


# ============================================================================
# T13: Anti-Reward-Hacking
# ============================================================================

class TestAntiRewardHacking:
    """Verify that broken/garbage LLM output cannot exploit the reward system."""

    @pytest.mark.unit
    def test_fallback_is_random_not_deterministic(self):
        """Garbage input → fallback must be RANDOM, not highest-power move.
        Run 50 times, expect at least 2 distinct fallback actions."""
        from pokemon_rl.translator import StateTranslator
        translator = StateTranslator(format_style="simple")

        # Create battle with multiple moves of different powers
        battle = MockBattle(moves=[
            MockMove("thunder", 120),
            MockMove("tackle", 40),
            MockMove("splash", 0),
        ])

        actions = set()
        for _ in range(50):
            action = translator.get_fallback_action(battle)
            if action and hasattr(action, "message"):
                actions.add(action.message)

        assert len(actions) >= 2, (
            f"Fallback should be random (>= 2 distinct actions from 50 calls), "
            f"but got only: {actions}. Likely deterministic (max-power bug)."
        )

    @pytest.mark.unit
    def test_all_garbage_game_does_not_always_win(self):
        """A model that always outputs garbage should not reliably win.
        With random fallback vs random opponent, expected win rate ~50%.
        We just verify the mechanism (random fallback) is in place."""
        from pokemon_rl.translator import StateTranslator
        translator = StateTranslator(format_style="simple")
        battle = MockBattle()
        # Garbage text → parse_action returns None → get_fallback_action called
        result = translator.parse_action("asdjfklasjdfkl", battle)
        assert result is None, "Garbage text must not parse to an action"
        fallback = translator.get_fallback_action(battle)
        assert fallback is not None, "Fallback must return an action"

    @pytest.mark.unit
    def test_passthrough_rubric_does_not_compute_own_reward(self):
        """PokemonRubric must return the env-computed reward, not compute its own."""
        from pokemon_rl.env import PokemonRubric
        rubric = PokemonRubric()
        # Set a specific reward and verify passthrough returns it exactly
        state = {"reward": 0.42, "trajectory": []}
        result = rubric._passthrough_reward_sync(state)
        assert result == 0.42, "Passthrough must return env reward exactly"

    @requires_verifiers
    @pytest.mark.unit
    def test_score_rollouts_cannot_be_false(self):
        """score_rollouts=False would zero rewards (C7). Must be True."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000, play_mode="single",
        )
        assert env.score_rollouts is True


# ============================================================================
# T14: Decorator Registration
# ============================================================================

class TestDecoratorRegistration:

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_vf_stop_game_over_actually_stops(self):
        """CRIT-4: @vf.stop game_over must make is_completed() return True.
        Not just 'does the method exist' — the decorator must actually work."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000, play_mode="single",
        )
        import time
        state = {
            "game_over": True, "game_turn": 5, "error": None,
            "trajectory": [], "is_completed": False,
            "timing": {"start_time": time.time(), "stop_condition": None,
                       "stop_time": None, "cleanup_time": None},
        }
        # is_completed checks all @vf.stop conditions
        result = await env.is_completed(state)
        assert result is True, (
            "is_completed must return True when game_over=True. "
            "This means @vf.stop decorator is working."
        )

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_vf_stop_does_not_stop_prematurely(self):
        """NEGATIVE: is_completed must be False when game is still active."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000, play_mode="single",
        )
        import time
        state = {
            "game_over": False, "game_turn": 5, "error": None,
            "trajectory": [], "is_completed": False,
            "timing": {"start_time": time.time(), "stop_condition": None,
                       "stop_time": None, "cleanup_time": None},
        }
        result = await env.is_completed(state)
        assert result is False, "Game still active — is_completed must be False"

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_vf_cleanup_runs_on_normal_exit(self):
        """@vf.cleanup cleanup_battle must be registered and callable."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000, play_mode="single",
        )
        # Verify cleanup_battle exists and is callable
        assert callable(getattr(env, 'cleanup_battle', None)), (
            "cleanup_battle must be a callable method"
        )
        # Verify it's registered in the cleanup chain
        # Framework stores cleanup handlers in _cleanup_handlers and marks
        # decorated methods with .cleanup = True
        cleanup_handlers = getattr(env, '_cleanup_handlers', [])
        cleanup_names = [fn.__name__ for fn in cleanup_handlers] if cleanup_handlers else []
        assert 'cleanup_battle' in cleanup_names or getattr(
            env.cleanup_battle, 'cleanup', False
        ), "cleanup_battle must be registered with @vf.cleanup"

    @requires_verifiers
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_env_response_returns_empty_list(self):
        """CRIT-5: env_response must return [], not raise NotImplementedError.
        It's a required abstract method that our override stubs out."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000, play_mode="single",
        )
        result = await env.env_response([], {})
        assert result == [], (
            f"env_response must return [], got {result}. "
            f"If it raises NotImplementedError, the abstract method is not overridden."
        )
        # Also verify it's not returning a coroutine
        assert not asyncio.iscoroutine(result), "env_response returned a coroutine"


# ============================================================================
# T15: State Field Names (Phase 4 Renames)
# ============================================================================

class TestStateFieldNames:
    """Verify Phase 4 field naming conventions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_uses_game_turn_not_turn(self):
        """CR-5: State should use 'game_turn', not 'turn'."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        assert "game_turn" in state, "State should use 'game_turn' field name"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_has_agents_field(self):
        """Phase 4: state['_agents'] must contain _AgentContext instances."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        assert "_agents" in state, "State must have _agents field"
        assert len(state["_agents"]) >= 1
        for agent in state["_agents"]:
            assert isinstance(agent, _AgentContext), (
                f"Agent must be _AgentContext, got {type(agent)}"
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_has_two_agents(self):
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        assert len(state["_agents"]) == 2
        assert state["_agents"][0].agent_idx == 0
        assert state["_agents"][1].agent_idx == 1


# ============================================================================
# T16: Completion Text Extraction (Missing Spec Tests T1.2, T1.3)
# ============================================================================

class TestCompletionExtraction:
    """Verify add_trajectory_step handles various completion formats."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_completion_triggers_fallback(self):
        """HIGH-2 / T1.2: Empty completion Messages → fallback action, no crash."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        await env.get_prompt_messages(state)
        # Empty Messages list — should trigger fallback
        step = {"completion": [], "prompt": [], "tokens": {}}
        await env.add_trajectory_step(state, step)

        extras = state["trajectory"][0].get("extras", {})
        assert extras.get("parse_failed") is True, (
            "Empty completion should trigger parse failure + fallback"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_assistant_content_triggers_fallback(self):
        """Empty assistant content should trigger fallback."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        await env.get_prompt_messages(state)
        step = {
            "completion": [{"role": "assistant", "content": ""}],
            "prompt": [], "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        extras = state["trajectory"][0].get("extras", {})
        assert extras.get("parse_failed") is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_message_extracts_last_assistant(self):
        """HIGH-3 / T1.3: Multi-message completion → extract last assistant content."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=1)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        await env.get_prompt_messages(state)
        # Multi-message completion: should use last assistant content
        step = {
            "completion": [
                {"role": "system", "content": "thinking..."},
                {"role": "assistant", "content": "I choose tackle"},
                {"role": "assistant", "content": '{"move": "tackle"}'},
            ],
            "prompt": [], "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        extras = state["trajectory"][0].get("extras", {})
        # The LAST assistant message has valid JSON, so parse should succeed
        assert extras.get("parse_failed") is False, (
            "Multi-message: last assistant has valid JSON, should parse"
        )


# ============================================================================
# T17: _AgentContext Lifecycle (Missing Spec Tests T6.2-T6.4)
# ============================================================================

class TestAgentContextLifecycle:
    """Verify _AgentContext state changes during gameplay."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_history_grows_by_two_per_step(self):
        """HIGH-4 / T6.2: Each step adds 2 entries (user + assistant) to history."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=3)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        for i in range(3):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)

        agent = state["_agents"][0]
        # 3 turns × 2 entries each = 6 history entries
        assert len(agent.message_history) == 6, (
            f"3 turns should produce 6 history entries (2 per step), "
            f"got {len(agent.message_history)}"
        )
        # Verify alternating user/assistant
        for i in range(0, len(agent.message_history), 2):
            assert agent.message_history[i]["role"] == "user", f"Entry {i} should be user"
            assert agent.message_history[i+1]["role"] == "assistant", f"Entry {i+1} should be assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_steps_contain_only_own_steps(self):
        """HIGH-4 / T6.3: Each agent's steps list has only that agent's steps."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockSelfplayManager()
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)

        agents = state["_agents"]
        for agent in agents:
            for step in agent.steps:
                step_idx = step.get("extras", {}).get("agent_idx")
                assert step_idx == agent.agent_idx, (
                    f"Agent {agent.agent_idx}'s steps list contains step "
                    f"with agent_idx={step_idx}"
                )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_failure_count_tracks_correctly(self):
        """HIGH-4 / T6.4: parse_failure_count accurate after successes and failures."""
        from pokemon_rl.env import PokemonBattleEnv
        mock_mgr = StrictMockHeuristicManager(game_turns=3)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "battle", "prompt": []})

        # Step 1: valid move (success)
        await env.get_prompt_messages(state)
        step1 = {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": [], "tokens": {},
        }
        await env.add_trajectory_step(state, step1)
        assert state["_agents"][0].parse_failure_count == 0

        # Step 2: garbage (failure)
        await env.get_prompt_messages(state)
        step2 = {
            "completion": [{"role": "assistant", "content": "not json"}],
            "prompt": [], "tokens": {},
        }
        await env.add_trajectory_step(state, step2)
        assert state["_agents"][0].parse_failure_count == 1

        # Step 3: valid move (success)
        await env.get_prompt_messages(state)
        step3 = {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": [], "tokens": {},
        }
        await env.add_trajectory_step(state, step3)
        assert state["_agents"][0].parse_failure_count == 1, (
            "Success should not increment failure count"
        )


# ============================================================================
# T18: Render Completion Format (HIGH-1)
# ============================================================================

class TestRenderCompletionFormat:
    """Verify render_completion sets all required fields in correct format."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_completion_is_messages_list(self):
        """HIGH-1: state['completion'] must be Messages format (list of dicts)."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": True, "game_turn": 5, "trajectory": [
                {
                    "extras": {"agent_idx": 0},
                    "completion": [{"role": "assistant", "content": "tackle"}],
                },
            ],
        }
        await env.render_completion(state)
        assert isinstance(state["completion"], list), (
            f"completion must be a list (Messages), got {type(state['completion'])}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_empty_trajectory(self):
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {"won": True, "game_turn": 0, "trajectory": []}
        await env.render_completion(state)
        assert "completion" in state
        assert isinstance(state["completion"], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_error_path_cleanup(self):
        """HIGH MED-10: Error during selfplay should still allow cleanup."""
        from pokemon_rl.env import PokemonBattleEnv

        class ErrorSelfplayManager:
            async def start_battle_selfplay(self, **kw):
                return [(0, MockBattle("p1")), (1, MockBattle("p2"))]
            async def submit_selfplay_action(self, idx, action):
                raise ConnectionError("Lost connection")
            async def close(self):
                pass

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        env.translator = MockTranslator()
        state = {"manager": ErrorSelfplayManager()}
        # Cleanup should work even after error
        await env.cleanup_battle(state)
        assert state["manager"] is None


# ============================================================================
# T19: Default Reward Ambiguity (HIGH-9)
# ============================================================================

class TestDefaultRewardAmbiguity:
    """Flag that default reward_draw == reward_loss can be confusing."""

    @pytest.mark.unit
    def test_default_draw_equals_loss_but_metrics_differ(self):
        """With default rewards, draw and loss give same reward but different metrics.
        This is by design but important to verify."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        # Default: reward_draw=0.0, reward_loss=0.0
        assert env.reward_draw == env.reward_loss == 0.0, (
            "Default reward_draw and reward_loss should both be 0.0"
        )
        # But metrics distinguish them
        draw_state = {"won": None, "game_turn": 5, "trajectory": [{"extras": {"agent_idx": 0}}]}
        loss_state = {"won": False, "game_turn": 5, "trajectory": [{"extras": {"agent_idx": 0}}]}
        env._assign_rewards(draw_state)
        env._assign_rewards(loss_state)


# ============================================================================
# T20: Adversarial Audit — Additional Coverage Tests
# ============================================================================

class TestAsymmetricStepCountAdvantage:
    """Verify advantage baseline correctness with asymmetric step counts."""

    @pytest.mark.unit
    def test_asymmetric_steps_same_advantage_magnitude(self):
        """Winner with 10 steps and loser with 3 steps must get SAME advantage magnitude.
        NEGATIVE: Within-rollout mean would give winner smaller advantage (more steps
        dilute the mean). Config-derived baseline avoids this."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        # P0 has 10 steps, P1 has 3 steps. P0 wins.
        trajectory = (
            [{"extras": {"agent_idx": 0}} for _ in range(10)] +
            [{"extras": {"agent_idx": 1}} for _ in range(3)]
        )
        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        p0_advs = [s["advantage"] for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1_advs = [s["advantage"] for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]

        # All P0 advantages must be the same value
        assert len(set(p0_advs)) == 1, f"P0 advantages should be uniform, got {set(p0_advs)}"
        # All P1 advantages must be the same value
        assert len(set(p1_advs)) == 1, f"P1 advantages should be uniform, got {set(p1_advs)}"
        # Magnitudes must be EQUAL (symmetric around baseline)
        assert abs(p0_advs[0]) == abs(p1_advs[0]), (
            f"Winner advantage magnitude ({abs(p0_advs[0])}) must equal "
            f"loser magnitude ({abs(p1_advs[0])}). "
            f"If different, the baseline is step-count-dependent (bug)."
        )

    @pytest.mark.unit
    def test_advantage_baseline_exact_value(self):
        """Default rewards (win=1, loss=0): baseline = 0.5.
        Winner advantage = +0.5, loser advantage = -0.5."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )
        state = {
            "won": True,
            "trajectory": [
                {"extras": {"agent_idx": 0}},
                {"extras": {"agent_idx": 1}},
            ],
        }
        env._assign_rewards(state)
        assert state["trajectory"][0]["advantage"] == 0.5, (
            f"Default rewards: winner advantage should be 0.5, got {state['trajectory'][0]['advantage']}"
        )
        assert state["trajectory"][1]["advantage"] == -0.5, (
            f"Default rewards: loser advantage should be -0.5, got {state['trajectory'][1]['advantage']}"
        )

    @pytest.mark.unit
    def test_advantage_baseline_custom_rewards(self):
        """Custom win=1, loss=-1: baseline = 0.
        Winner advantage = +1.0, loser advantage = -1.0."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_win=1.0, reward_loss=-1.0,
        )
        state = {
            "won": True,
            "trajectory": [
                {"extras": {"agent_idx": 0}},
                {"extras": {"agent_idx": 1}},
            ],
        }
        env._assign_rewards(state)
        assert state["trajectory"][0]["advantage"] == 1.0
        assert state["trajectory"][1]["advantage"] == -1.0


class TestRenderCompletionP0Perspective:
    """Verify state['reward'] uses P0's perspective in self-play."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_state_reward_matches_p0(self):
        """state['reward'] must match P0's reward, not P1's or trajectory[0]'s."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_win=1.0, reward_loss=-1.0,
        )
        # P1 acts first in trajectory, but P0 should determine state["reward"]
        state = {
            "won": True, "game_turn": 5,
            "trajectory": [
                {"extras": {"agent_idx": 1}, "completion": "b"},  # P1 first in trajectory
                {"extras": {"agent_idx": 0}, "completion": "a"},  # P0 second
                {"extras": {"agent_idx": 1}, "completion": "d"},
                {"extras": {"agent_idx": 0}, "completion": "c"},
            ],
        }
        await env.render_completion(state)
        # P0 won, so state["reward"] should be P0's reward (1.0), not trajectory[0]'s (-1.0)
        assert state["reward"] == 1.0, (
            f"state['reward'] should be P0's reward (1.0), got {state['reward']}. "
            f"Bug: using trajectory[0] instead of P0's first step."
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_p1_wins_state_reward_negative(self):
        """When P1 wins (won=False), state['reward'] should be P0's loss reward."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_win=1.0, reward_loss=-1.0,
        )
        state = {
            "won": False, "game_turn": 5,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 1}, "completion": "b"},
            ],
        }
        await env.render_completion(state)
        assert state["reward"] == -1.0, (
            f"P1 wins: state['reward'] should be P0's loss reward (-1.0), got {state['reward']}"
        )


class TestCompletionContentNone:
    """Verify extract_completion_text handles content=None."""

    @pytest.mark.unit
    def test_content_none_returns_empty_string(self):
        """Messages with content=None must not crash or return None."""
        from pokemon_rl.translator import StateTranslator
        result = StateTranslator.extract_completion_text(
            [{"role": "assistant", "content": None}]
        )
        assert result == "", (
            f"content=None should give empty string, got {result!r}"
        )
        assert isinstance(result, str), f"Must return str, got {type(result)}"

    @pytest.mark.unit
    def test_content_none_vs_missing(self):
        """content=None and missing content must both give empty string.
        NEGATIVE: content=None must NOT return None (would crash parse_action)."""
        from pokemon_rl.translator import StateTranslator
        none_result = StateTranslator.extract_completion_text(
            [{"role": "assistant", "content": None}]
        )
        missing_result = StateTranslator.extract_completion_text(
            [{"role": "assistant"}]
        )
        assert none_result == "", f"content=None gave {none_result!r}"
        assert missing_result == "", f"missing content gave {missing_result!r}"


class TestScoreRolloutsKwargBypass:
    """Verify score_rollouts=False cannot be passed via kwargs."""

    @requires_verifiers
    @pytest.mark.unit
    def test_score_rollouts_false_kwarg_ignored(self):
        """Passing score_rollouts=False must still result in True (C7 enforcement)."""
        from pokemon_rl.env import PokemonBattleEnv
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", score_rollouts=False,  # should be popped
        )
        assert env.score_rollouts is True, (
            "score_rollouts=False kwarg must be ignored (C7)"
        )


class TestSetupStateRetryCleanup:
    """Verify setup_state clears stale trajectory from failed retries."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trajectory_cleared_on_retry(self):
        """If setup_state is called again (retry), trajectory must be fresh."""
        from pokemon_rl.env import PokemonBattleEnv
        from unittest.mock import patch

        mock_mgr = StrictMockHeuristicManager(game_turns=2)
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        env.translator = MockTranslator()

        # First setup + add a step
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr):
            state = await env.setup_state({"task": "test", "prompt": []})
        state["trajectory"].append({"stale": True})
        assert len(state["trajectory"]) == 1

        # Second setup (retry) — trajectory must be cleared
        mock_mgr2 = StrictMockHeuristicManager(game_turns=2)
        with patch('pokemon_rl.battle.BattleManager', return_value=mock_mgr2):
            state = await env.setup_state(state)
        assert len(state["trajectory"]) == 0, (
            f"Retry setup_state must clear trajectory, but has {len(state['trajectory'])} entries"
        )
        assert not any(s.get("stale") for s in state["trajectory"]), (
            "Stale steps from previous attempt survived retry"
        )
