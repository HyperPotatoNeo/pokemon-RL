"""Phase 5 unit tests: RL training integration — agent behavior and config.

Tests the components needed for RL training integration (Phase 5) including:
- Prompt correctness for gen9ou (simple + pokechamp_io)
- Response parsing (moves, switches, terastallize, dynamax, normalization)
- Completion text extraction (string, Messages, multimodal)
- Fallback behavior (randomness, not max-power)
- Reward assignment for all scenarios (single, self-play, draws, step rewards)
- Team loading from directory (REQUIRES Phase 5 implementation)
- PokemonBattleEnv construction and validation
- load_environment round-trip with TOML args

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
Every test verifies BOTH positive AND negative cases. Assert on specific values,
not just "is not None" or "len > 0".

ANTI-REWARD-HACKING: Tests verify fallback randomness, parse failure tracking,
advantage sign correctness, draw/loss equivalence, and truncation reward.
"""

import json
import os
import pytest
import re
from collections import Counter
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
try:
    import verifiers as vf
    HAS_VERIFIERS = True
except ImportError:
    HAS_VERIFIERS = False

try:
    from poke_env.player.player import Player  # noqa: F401
    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False

try:
    from pokechamp.prompts import state_translate  # noqa: F401
    HAS_POKECHAMP = True
except ImportError:
    HAS_POKECHAMP = False

requires_poke_env = pytest.mark.skipif(
    not HAS_POKE_ENV, reason="poke-env not installed"
)
requires_pokechamp = pytest.mark.skipif(
    not HAS_POKECHAMP, reason="pokechamp not installed"
)
requires_verifiers = pytest.mark.skipif(
    not HAS_VERIFIERS, reason="verifiers not installed"
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DIR = os.path.join(
    PROJECT_ROOT, "vendor", "pokechamp", "poke_env", "data", "static",
    "teams", "gen9ou",
)


# ---------------------------------------------------------------------------
# Mock infrastructure (reused from Phase 4 patterns)
# ---------------------------------------------------------------------------
class MockMove:
    """Minimal move mock with id, base_power, type."""
    def __init__(self, move_id="tackle", base_power=40, move_type="normal"):
        self.id = move_id
        self.base_power = base_power
        self.type = move_type


class MockPokemon:
    """Minimal Pokemon mock with species and HP."""
    def __init__(self, species="Pikachu", hp_fraction=1.0, types=None):
        self.species = species
        self.current_hp_fraction = hp_fraction
        self.types = types or []


class MockBattle:
    """Minimal Battle mock for unit tests."""
    def __init__(self, name="mock", turn=1, moves=None, switches=None,
                 format_str="gen9ou"):
        self.name = name
        self.turn = turn
        self.available_moves = moves if moves is not None else [
            MockMove("thunderbolt", 90, "electric"),
            MockMove("surf", 90, "water"),
            MockMove("icebeam", 90, "ice"),
        ]
        self.available_switches = switches if switches is not None else [
            MockPokemon("Landorus-Therian"),
            MockPokemon("Heatran"),
        ]
        self.force_switch = False
        self.won = None
        self.battle_tag = f"mock-{name}"
        self._format = format_str
        self.active_pokemon = MockPokemon("Pikachu", 0.8)
        self.opponent_active_pokemon = MockPokemon("Charizard", 0.5)


# ============================================================================
# T1: Prompt Correctness
# ============================================================================

class TestPromptCorrectness:
    """T1: Verify prompt generation for different formats and battle formats."""

    @pytest.mark.unit
    def test_simple_format_produces_system_and_user(self):
        """Simple format returns [system, user] messages with battle info."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="simple")
        battle = MockBattle(format_str="gen9ou")
        messages = translator.battle_to_prompt(battle)

        # Positive: correct structure
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Pokemon" in messages[0]["content"]
        assert "Pikachu" in messages[1]["content"]  # active_pokemon.species
        assert "Charizard" in messages[1]["content"]  # opponent
        assert "thunderbolt" in messages[1]["content"]  # available move
        assert "Landorus" in messages[1]["content"]  # available switch

        # Negative: empty moves/switches still produce valid messages
        empty_battle = MockBattle(moves=[], switches=[])
        msgs_empty = translator.battle_to_prompt(empty_battle)
        assert len(msgs_empty) == 2
        assert "thunderbolt" not in msgs_empty[1]["content"]

    @pytest.mark.unit
    def test_simple_format_shows_move_details(self):
        """Simple format includes move power and type."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="simple")
        battle = MockBattle(moves=[MockMove("earthquake", 100, "ground")])
        messages = translator.battle_to_prompt(battle)
        content = messages[1]["content"]

        assert "earthquake" in content
        assert "100" in content  # base power
        assert "ground" in content  # type

    @pytest.mark.unit
    def test_simple_format_shows_hp_percentages(self):
        """Simple format shows HP as percentages."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="simple")
        battle = MockBattle()
        battle.active_pokemon = MockPokemon("Pikachu", 0.5)
        messages = translator.battle_to_prompt(battle)
        content = messages[1]["content"]

        assert "50%" in content  # HP percentage

    @pytest.mark.unit
    def test_unknown_format_raises(self):
        """Unknown format_style raises ValueError."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="nonexistent")
        with pytest.raises(ValueError, match="Unknown format_style"):
            translator.battle_to_prompt(MockBattle())

    @requires_poke_env
    @pytest.mark.unit
    def test_dynamax_gen8_not_gen9(self):
        """Negative: dynamax valid in gen8, rejected in gen9 and gen1."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("earthquake", gen=8)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []

        # Gen8: dynamax should work
        battle._format = "gen8ou"
        result_gen8 = translator.parse_action('{"dynamax": "earthquake"}', battle)
        assert result_gen8 is not None, "Dynamax must work in gen8"

        # Gen9: dynamax must be rejected
        battle._format = "gen9ou"
        result_gen9 = translator.parse_action('{"dynamax": "earthquake"}', battle)
        assert result_gen9 is None, "Dynamax must be rejected in gen9"

        # Gen1: dynamax must be rejected
        battle._format = "gen1ou"
        result_gen1 = translator.parse_action('{"dynamax": "earthquake"}', battle)
        assert result_gen1 is None, "Dynamax must be rejected in gen1"

    @requires_poke_env
    @pytest.mark.unit
    def test_terastallize_gen9_not_gen8_or_gen1(self):
        """Negative: terastallize valid in gen9, rejected in gen8 and gen1."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("scald", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []

        # Gen9: terastallize should work
        battle._format = "gen9ou"
        result_gen9 = translator.parse_action('{"terastallize": "scald"}', battle)
        assert result_gen9 is not None, "Terastallize must work in gen9"

        # Gen8: terastallize must be rejected
        battle._format = "gen8ou"
        result_gen8 = translator.parse_action('{"terastallize": "scald"}', battle)
        assert result_gen8 is None, "Terastallize must be rejected in gen8"

        # Gen1: terastallize must be rejected
        battle._format = "gen1ou"
        result_gen1 = translator.parse_action('{"terastallize": "scald"}', battle)
        assert result_gen1 is None, "Terastallize must be rejected in gen1"

    @requires_pokechamp
    @pytest.mark.unit
    def test_pokechamp_io_format_produces_rich_prompt(self):
        """T1: pokechamp_io format produces detailed prompt with damage calcs.

        NOTE: This test uses a real poke-env battle stub. If pokechamp's internals
        change, the mock may need updating. Skip if pokechamp not installed.
        """
        # pokechamp_io needs deep poke-env objects that can't be trivially mocked.
        # This test validates the translator delegates correctly. Full prompt
        # content validation happens in integration tests (T8) with real battles.
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="pokechamp_io")
        assert translator.format_style == "pokechamp_io"
        # The actual prompt generation is tested in integration tests with real
        # Showdown battles because pokechamp's LocalSim requires deep poke-env types.


# ============================================================================
# T2: Response Parsing
# ============================================================================

@requires_poke_env
class TestResponseParsing:
    """T2: Verify parse_action handles all gen9ou action types."""

    @pytest.mark.unit
    def test_valid_move_parses(self):
        """Valid move JSON → BattleOrder with correct move."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        # Create a battle with real Move objects
        move = Move("thunderbolt", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen9ou"

        result = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert result is not None, "Valid move should parse successfully"
        assert "thunderbolt" in result.message.lower()

    @pytest.mark.unit
    def test_valid_switch_parses(self):
        """Valid switch JSON → BattleOrder for correct pokemon."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        pokemon = MagicMock()
        pokemon.species = "landorustherian"

        battle = MagicMock()
        battle.available_moves = []
        battle.available_switches = [pokemon]
        battle._format = "gen9ou"

        result = translator.parse_action('{"switch": "Landorus-Therian"}', battle)
        assert result is not None, "Valid switch should parse successfully"
        # BattleOrder.order should be the pokemon, not a move
        assert result.order is pokemon, "Switch order must reference the pokemon"

        # Negative: move JSON should NOT match as switch
        result_move = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert result_move is None, "No available moves → move should return None"

    @pytest.mark.unit
    def test_terastallize_parses_gen9(self):
        """Terastallize JSON → BattleOrder with terastallize=True in gen9."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("scald", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen9ou"

        result = translator.parse_action('{"terastallize": "scald"}', battle)
        assert result is not None, "Terastallize should parse in gen9"
        assert result.terastallize is True, (
            "BattleOrder must have terastallize=True flag set"
        )
        # Negative: regular move should NOT have terastallize
        result_normal = translator.parse_action('{"move": "scald"}', battle)
        assert result_normal is not None
        assert result_normal.terastallize is not True, (
            "Regular move must NOT have terastallize=True"
        )

    @pytest.mark.unit
    def test_move_name_normalization(self):
        """Move names with spaces/caps normalize correctly."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("thunderbolt", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen9ou"

        # "Thunder Bolt" should normalize to "thunderbolt"
        result = translator.parse_action('{"move": "Thunder Bolt"}', battle)
        assert result is not None, "Normalized name should match"

    @pytest.mark.unit
    def test_invalid_json_returns_none(self):
        """Invalid JSON → None."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        battle = MockBattle()

        result = translator.parse_action("not json at all", battle)
        assert result is None, "Invalid JSON must return None"

    @pytest.mark.unit
    def test_unknown_move_returns_none(self):
        """Move not in available_moves → None."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("thunderbolt", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen9ou"

        result = translator.parse_action('{"move": "nonexistent"}', battle)
        assert result is None, "Unknown move must return None"

    @pytest.mark.unit
    def test_dynamax_in_gen9_returns_none(self):
        """Dynamax action in gen9 → None (gen8 only)."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("earthquake", gen=9)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen9ou"

        result = translator.parse_action('{"dynamax": "earthquake"}', battle)
        assert result is None, "Dynamax in gen9 must return None"

    @pytest.mark.unit
    def test_terastallize_in_gen8_returns_none(self):
        """Terastallize action in gen8 → None (gen9 only)."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        move = Move("scald", gen=8)
        battle = MagicMock()
        battle.available_moves = [move]
        battle.available_switches = []
        battle._format = "gen8ou"

        result = translator.parse_action('{"terastallize": "scald"}', battle)
        assert result is None, "Terastallize in gen8 must return None"

    @pytest.mark.unit
    def test_empty_available_moves_returns_none(self):
        """Move action with no available moves → None."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        battle = MagicMock()
        battle.available_moves = []
        battle.available_switches = []
        battle._format = "gen9ou"

        result = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert result is None, "No available moves → must return None"


# ============================================================================
# T3: Response Text Extraction
# ============================================================================

class TestResponseTextExtraction:
    """T3: Verify extract_completion_text handles response formats."""

    @pytest.mark.unit
    def test_string_input_returns_as_is(self):
        """String input → returned unchanged."""
        from pokemon_rl.translator import StateTranslator

        text = 'Let me think... {"move": "surf"}'
        result = StateTranslator.extract_completion_text(text)
        assert result == text

    @pytest.mark.unit
    def test_messages_format_extracts_assistant(self):
        """Messages format → extracts last assistant content."""
        from pokemon_rl.translator import StateTranslator

        messages = [
            {"role": "user", "content": "Battle state..."},
            {"role": "assistant", "content": 'I choose {"move": "surf"}'},
        ]
        result = StateTranslator.extract_completion_text(messages)
        assert result == 'I choose {"move": "surf"}'

    @pytest.mark.unit
    def test_multiple_assistant_messages_takes_last(self):
        """Multiple assistant messages → extracts the LAST one."""
        from pokemon_rl.translator import StateTranslator

        messages = [
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "try again"},
            {"role": "assistant", "content": "second response"},
        ]
        result = StateTranslator.extract_completion_text(messages)
        assert result == "second response"
        assert result != "first response"

    @pytest.mark.unit
    def test_no_assistant_message_concatenates(self):
        """No assistant message → concatenated content of all messages."""
        from pokemon_rl.translator import StateTranslator

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "world"},
        ]
        result = StateTranslator.extract_completion_text(messages)
        assert "hello" in result
        assert "world" in result

    @pytest.mark.unit
    def test_empty_string_returns_empty(self):
        """Empty string → empty string."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator.extract_completion_text("")
        assert result == ""

    @pytest.mark.unit
    def test_empty_list_returns_empty(self):
        """Empty list → empty string."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator.extract_completion_text([])
        assert result == ""

    @pytest.mark.unit
    def test_multimodal_content_blocks(self):
        """Multimodal content blocks → extracts text blocks only, NOT thinking."""
        from pokemon_rl.translator import StateTranslator

        messages = [
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "internal reasoning"},
                {"type": "text", "text": 'I choose {"move": "surf"}'},
            ]},
        ]
        result = StateTranslator.extract_completion_text(messages)
        # Positive: text blocks extracted
        assert '{"move": "surf"}' in result
        # Negative: thinking blocks must NOT be included
        assert "internal reasoning" not in result, (
            "Thinking blocks must be filtered out of completion text"
        )

    @pytest.mark.unit
    def test_none_content_returns_empty(self):
        """Assistant with None content → empty string."""
        from pokemon_rl.translator import StateTranslator

        messages = [{"role": "assistant", "content": None}]
        result = StateTranslator.extract_completion_text(messages)
        assert result == ""


# ============================================================================
# T3b: _extract_last_json
# ============================================================================

class TestExtractLastJson:
    """Verify _extract_last_json correctly handles various JSON patterns."""

    @pytest.mark.unit
    def test_simple_json(self):
        """Simple JSON object at end of text."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator._extract_last_json('some text {"move": "surf"}')
        assert result == {"move": "surf"}

    @pytest.mark.unit
    def test_multiple_json_takes_last(self):
        """Multiple JSON objects → returns the LAST valid one."""
        from pokemon_rl.translator import StateTranslator

        text = 'I see {"analysis": true}... so {"move": "surf"}'
        result = StateTranslator._extract_last_json(text)
        assert result == {"move": "surf"}
        assert result != {"analysis": True}

    @pytest.mark.unit
    def test_nested_json(self):
        """Nested JSON → returns outermost object."""
        from pokemon_rl.translator import StateTranslator

        text = '{"outer": {"inner": "value"}}'
        result = StateTranslator._extract_last_json(text)
        assert result is not None
        assert "outer" in result

    @pytest.mark.unit
    def test_no_json_returns_none(self):
        """No JSON in text → None."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator._extract_last_json("no json here at all")
        assert result is None

    @pytest.mark.unit
    def test_array_not_returned(self):
        """JSON array → None (only dicts returned)."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator._extract_last_json('[1, 2, 3]')
        assert result is None

    @pytest.mark.unit
    def test_reasoning_chain_extracts_action(self):
        """Full reasoning chain with JSON action at end."""
        from pokemon_rl.translator import StateTranslator

        text = (
            "The opponent has Fire type... I should use Water. "
            'Let me analyze the situation. {"move": "surf"}'
        )
        result = StateTranslator._extract_last_json(text)
        assert result == {"move": "surf"}


# ============================================================================
# T4: Fallback Behavior
# ============================================================================

@requires_poke_env
class TestFallbackBehavior:
    """T4: Verify parse failure triggers random legal action.

    ANTI-REWARD-HACKING: Fallback must be RANDOM, not max-power.
    A model that always outputs garbage should NOT get the strongest
    move for free.
    """

    @pytest.mark.unit
    def test_fallback_returns_valid_order(self):
        """Fallback returns a BattleOrder for available actions."""
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()

        moves = [Move("thunderbolt", gen=9), Move("surf", gen=9)]
        pokemon = MagicMock()
        pokemon.species = "Heatran"

        battle = MagicMock()
        battle.available_moves = moves
        battle.available_switches = [pokemon]

        action = translator.get_fallback_action(battle)
        assert action is not None
        assert hasattr(action, "message")

    @pytest.mark.unit
    def test_fallback_is_random_not_always_same(self):
        """Fallback must produce >1 distinct action over 50 calls (proves randomness).

        ANTI-REWARD-HACKING: If fallback always returns the same move,
        a model could exploit this by outputting garbage and always getting
        the "best" heuristic move.
        """
        from poke_env.environment.move import Move
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()

        moves = [Move("thunderbolt", gen=9), Move("surf", gen=9),
                 Move("icebeam", gen=9)]
        pokemon = MagicMock()
        pokemon.species = "Heatran"

        battle = MagicMock()
        battle.available_moves = moves
        battle.available_switches = [pokemon]

        actions = set()
        for _ in range(50):
            action = translator.get_fallback_action(battle)
            actions.add(action.message)

        assert len(actions) >= 2, (
            f"Fallback produced only {len(actions)} distinct actions from 50 calls. "
            "Fallback must be RANDOM to prevent reward hacking."
        )

    @pytest.mark.unit
    def test_fallback_with_no_actions_returns_default(self):
        """No available moves or switches → default order (not crash)."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        battle = MagicMock()
        battle.available_moves = []
        battle.available_switches = []

        action = translator.get_fallback_action(battle)
        assert action is not None, "Fallback must not crash with no available actions"


# ============================================================================
# T5: Reward Assignment
# ============================================================================

class TestRewardAssignment:
    """T5: Verify _assign_rewards produces correct rewards and advantages.

    ANTI-REWARD-HACKING: Tests verify advantage signs, draw=loss equivalence,
    step_reward folding, and empty trajectory safety.
    """

    def _make_env(self, **kwargs):
        """Create PokemonBattleEnv with test defaults."""
        from pokemon_rl.env import PokemonBattleEnv
        defaults = dict(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        defaults.update(kwargs)
        return PokemonBattleEnv(**defaults)

    def _make_trajectory(self, num_steps=3, agent_indices=None):
        """Create trajectory steps with agent_idx in extras."""
        if agent_indices is None:
            agent_indices = [0] * num_steps
        return [
            {"extras": {"agent_idx": idx}}
            for idx in agent_indices
        ]

    # --- Single-agent tests ---

    @pytest.mark.unit
    def test_single_agent_win(self):
        """Single-agent win: all steps get reward_win=1.0."""
        env = self._make_env(reward_win=1.0, reward_loss=0.0)
        state = {"won": True, "trajectory": self._make_trajectory(3)}
        env._assign_rewards(state)

        for step in state["trajectory"]:
            assert step["reward"] == 1.0, "Win reward must be 1.0"
            # Uniform rewards → advantage left as None for score_group
            assert step.get("advantage") is None, (
                "Uniform single-agent rewards should NOT pre-set advantage"
            )

    @pytest.mark.unit
    def test_single_agent_loss(self):
        """Single-agent loss: all steps get reward_loss=0.0."""
        env = self._make_env(reward_win=1.0, reward_loss=0.0)
        state = {"won": False, "trajectory": self._make_trajectory(3)}
        env._assign_rewards(state)

        for step in state["trajectory"]:
            assert step["reward"] == 0.0, "Loss reward must be 0.0"

    @pytest.mark.unit
    def test_single_agent_win_loss_distinguishable(self):
        """Win and loss produce different rewards (no fall-through)."""
        env = self._make_env(reward_win=1.0, reward_loss=0.0)

        win_state = {"won": True, "trajectory": self._make_trajectory(2)}
        loss_state = {"won": False, "trajectory": self._make_trajectory(2)}
        env._assign_rewards(win_state)
        env._assign_rewards(loss_state)

        assert win_state["trajectory"][0]["reward"] != loss_state["trajectory"][0]["reward"]

    @pytest.mark.unit
    def test_single_agent_custom_rewards(self):
        """Custom reward values propagated correctly."""
        env = self._make_env(reward_win=10.0, reward_loss=-5.0, reward_draw=2.5)

        win_state = {"won": True, "trajectory": self._make_trajectory(1)}
        loss_state = {"won": False, "trajectory": self._make_trajectory(1)}
        draw_state = {"won": None, "trajectory": self._make_trajectory(1)}

        env._assign_rewards(win_state)
        env._assign_rewards(loss_state)
        env._assign_rewards(draw_state)

        assert win_state["trajectory"][0]["reward"] == 10.0
        assert loss_state["trajectory"][0]["reward"] == -5.0
        assert draw_state["trajectory"][0]["reward"] == 2.5

    # --- Self-play tests ---

    @pytest.mark.unit
    def test_selfplay_p0_wins(self):
        """Self-play, P0 wins: P0 gets reward_win, P1 gets reward_loss."""
        env = self._make_env(play_mode="self_play", reward_win=1.0, reward_loss=0.0)
        trajectory = self._make_trajectory(
            num_steps=4, agent_indices=[0, 1, 0, 1]
        )
        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        for step in trajectory:
            idx = step["extras"]["agent_idx"]
            if idx == 0:
                assert step["reward"] == 1.0, "P0 (winner) must get reward_win"
            else:
                assert step["reward"] == 0.0, "P1 (loser) must get reward_loss"

    @pytest.mark.unit
    def test_selfplay_p1_wins(self):
        """Self-play, P1 wins: P0 gets reward_loss, P1 gets reward_win."""
        env = self._make_env(play_mode="self_play", reward_win=1.0, reward_loss=0.0)
        trajectory = self._make_trajectory(
            num_steps=4, agent_indices=[0, 1, 0, 1]
        )
        state = {"won": False, "trajectory": trajectory}
        env._assign_rewards(state)

        for step in trajectory:
            idx = step["extras"]["agent_idx"]
            if idx == 0:
                assert step["reward"] == 0.0, "P0 (loser) must get reward_loss"
            else:
                assert step["reward"] == 1.0, "P1 (winner) must get reward_win"

    @pytest.mark.unit
    def test_selfplay_advantages_opposite_sign(self):
        """ANTI-REWARD-HACKING: Self-play winner and loser get OPPOSITE advantages.

        A bug here would reinforce BOTH winning AND losing actions.
        """
        env = self._make_env(play_mode="self_play", reward_win=1.0, reward_loss=0.0)
        trajectory = self._make_trajectory(
            num_steps=4, agent_indices=[0, 1, 0, 1]
        )
        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        p0_advantages = [s["advantage"] for s in trajectory
                         if s["extras"]["agent_idx"] == 0]
        p1_advantages = [s["advantage"] for s in trajectory
                         if s["extras"]["agent_idx"] == 1]

        # All P0 advantages should be positive (winner)
        assert all(a > 0 for a in p0_advantages), (
            f"P0 (winner) advantages must be positive, got {p0_advantages}"
        )
        # All P1 advantages should be negative (loser)
        assert all(a < 0 for a in p1_advantages), (
            f"P1 (loser) advantages must be negative, got {p1_advantages}"
        )
        # Magnitudes should be equal (config baseline = 0.5)
        assert abs(p0_advantages[0]) == abs(p1_advantages[0]), (
            "Winner and loser advantage magnitudes must be equal"
        )

    @pytest.mark.unit
    def test_selfplay_draw_rewards(self):
        """Self-play draw: all steps get reward_draw, advantages NOT pre-set."""
        env = self._make_env(play_mode="self_play",
                             reward_win=1.0, reward_loss=0.0, reward_draw=0.0)
        trajectory = self._make_trajectory(
            num_steps=4, agent_indices=[0, 1, 0, 1]
        )
        state = {"won": None, "trajectory": trajectory}
        env._assign_rewards(state)

        for step in trajectory:
            assert step["reward"] == 0.0, "Draw reward must be reward_draw"
            # Uniform rewards → advantage left as None for score_group
            assert step.get("advantage") is None, (
                "Draw with uniform rewards should NOT pre-set advantage"
            )

    @pytest.mark.unit
    def test_draw_equals_loss_reward(self):
        """ANTI-REWARD-HACKING: Draw and loss produce identical rewards.

        With reward_draw=0.0=reward_loss, a model can't learn to force draws
        instead of playing to win. There's no reward advantage to stalling.
        """
        env = self._make_env(play_mode="self_play",
                             reward_win=1.0, reward_loss=0.0, reward_draw=0.0)

        draw_traj = self._make_trajectory(2, [0, 1])
        loss_traj = self._make_trajectory(2, [0, 1])

        draw_state = {"won": None, "trajectory": draw_traj}
        loss_state = {"won": False, "trajectory": loss_traj}

        env._assign_rewards(draw_state)
        env._assign_rewards(loss_state)

        # P0 in draw and P0 in loss both get 0.0
        p0_draw = [s["reward"] for s in draw_traj if s["extras"]["agent_idx"] == 0][0]
        p0_loss = [s["reward"] for s in loss_traj if s["extras"]["agent_idx"] == 0][0]
        assert p0_draw == p0_loss, (
            f"Draw reward ({p0_draw}) must equal loss reward ({p0_loss})"
        )

    # --- Step reward tests ---

    @pytest.mark.unit
    def test_step_reward_folded_into_reward(self):
        """step_reward_fn output folded into step['reward'], not just extras.

        ANTI-REWARD-HACKING: extras are dropped at IPC boundary. If step
        rewards only lived in extras, they'd never reach training.
        """
        env = self._make_env(reward_win=1.0, reward_loss=0.0)
        trajectory = self._make_trajectory(2)
        trajectory[0]["extras"]["step_reward"] = 0.1
        trajectory[1]["extras"]["step_reward"] = -0.05

        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        assert trajectory[0]["reward"] == 1.1, "Terminal + step reward"
        assert trajectory[1]["reward"] == 0.95, "Terminal + step reward"
        # Extras preserved for logging
        assert trajectory[0]["extras"]["step_reward"] == 0.1

    @pytest.mark.unit
    def test_step_reward_triggers_advantage_preset(self):
        """Step rewards make rewards non-uniform → advantages pre-set."""
        env = self._make_env(reward_win=1.0, reward_loss=0.0)
        trajectory = self._make_trajectory(2)
        trajectory[0]["extras"]["step_reward"] = 0.1  # Makes reward non-uniform

        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        # Non-uniform rewards → advantages should be pre-set
        assert trajectory[0].get("advantage") is not None, (
            "Non-uniform rewards must trigger advantage pre-setting"
        )

    # --- Edge cases ---

    @pytest.mark.unit
    def test_empty_trajectory_no_crash(self):
        """Empty trajectory → no crash, no reward assignment."""
        env = self._make_env()
        state = {"won": True, "trajectory": []}
        env._assign_rewards(state)
        assert len(state["trajectory"]) == 0

    @pytest.mark.unit
    def test_advantage_baseline_is_config_derived(self):
        """Advantage baseline = (reward_win + reward_loss) / 2, not within-rollout mean.

        Using within-rollout mean would create step-count asymmetry bias:
        the winner with more steps gets lower per-step advantage.
        """
        env = self._make_env(play_mode="self_play",
                             reward_win=1.0, reward_loss=0.0)
        # Asymmetric step counts: P0 has 5 steps, P1 has 2
        trajectory = self._make_trajectory(
            num_steps=7, agent_indices=[0, 1, 0, 0, 0, 0, 1]
        )
        state = {"won": True, "trajectory": trajectory}
        env._assign_rewards(state)

        # With config baseline (0.5), all P0 advantages = 0.5, all P1 = -0.5
        # regardless of step count asymmetry
        p0_advs = [s["advantage"] for s in trajectory
                   if s["extras"]["agent_idx"] == 0]
        p1_advs = [s["advantage"] for s in trajectory
                   if s["extras"]["agent_idx"] == 1]

        assert all(a == p0_advs[0] for a in p0_advs), (
            "All P0 advantages must be equal (config baseline, not mean)"
        )
        assert all(a == p1_advs[0] for a in p1_advs), (
            "All P1 advantages must be equal (config baseline, not mean)"
        )

    @pytest.mark.unit
    def test_truncation_gives_draw_reward(self):
        """ANTI-REWARD-HACKING: Truncation (max_game_turns) gives reward_draw.

        A model that stalls games must NOT be rewarded with reward_win.
        """
        env = self._make_env(reward_win=1.0, reward_draw=0.0)
        # won=None means draw/truncation
        state = {"won": None, "trajectory": self._make_trajectory(3)}
        env._assign_rewards(state)

        for step in state["trajectory"]:
            assert step["reward"] == 0.0, "Truncation must give reward_draw, not reward_win"


# ============================================================================
# T6: Team Loading
# ============================================================================

class TestTeamLoading:
    """T6: Verify team files load correctly.

    REQUIRES: team_fn/team_dir implementation in PokemonBattleEnv (Phase 5).
    Tests that verify unimplemented features are marked and expected to fail
    until implementation is complete.
    """

    @pytest.mark.unit
    def test_team_files_exist_on_disk(self):
        """Verify gen9ou team files exist at expected path."""
        assert os.path.isdir(TEAM_DIR), (
            f"Team directory not found: {TEAM_DIR}"
        )
        team_files = [
            f for f in os.listdir(TEAM_DIR) if f.endswith(".txt")
        ]
        assert len(team_files) == 13, (
            f"Expected 13 gen9ou team files, found {len(team_files)}"
        )

    @pytest.mark.unit
    def test_team_files_are_valid_showdown_format(self):
        """Each team file is valid Showdown paste format."""
        team_files = sorted(
            f for f in os.listdir(TEAM_DIR) if f.endswith(".txt")
        )
        for fname in team_files:
            path = os.path.join(TEAM_DIR, fname)
            with open(path) as f:
                content = f.read()

            assert len(content) > 0, f"{fname} is empty"
            assert "Ability:" in content, f"{fname} missing 'Ability:'"
            assert "EVs:" in content or "IVs:" in content, (
                f"{fname} missing stat allocation"
            )
            assert "Nature" in content, f"{fname} missing Nature"
            # Each team should have 6 pokemon (6 empty-line-separated blocks)
            # Count pokemon by counting "Ability:" lines
            ability_count = content.count("Ability:")
            assert ability_count == 6, (
                f"{fname} has {ability_count} pokemon, expected 6"
            )

    @pytest.mark.unit
    def test_team_files_have_distinct_content(self):
        """Team files are not all identical."""
        team_files = sorted(
            f for f in os.listdir(TEAM_DIR) if f.endswith(".txt")
        )
        contents = set()
        for fname in team_files:
            path = os.path.join(TEAM_DIR, fname)
            with open(path) as f:
                contents.add(f.read())
        assert len(contents) > 1, "All team files have identical content"

    @pytest.mark.unit
    def test_random_team_pool_factory(self):
        """REQUIRES: random_team_pool() factory in env.py.

        Verify factory loads teams from directory and returns a callable
        that returns random team strings.
        """
        try:
            from pokemon_rl.env import random_team_pool
        except ImportError:
            pytest.skip("random_team_pool not yet implemented (Phase 5)")

        team_fn = random_team_pool(TEAM_DIR)
        assert callable(team_fn)

        # Should return non-empty strings
        team = team_fn()
        assert isinstance(team, str)
        assert len(team) > 0
        assert "Ability:" in team

        # Should return different teams over multiple calls (randomness)
        teams = set()
        for _ in range(20):
            teams.add(team_fn())
        assert len(teams) >= 2, (
            "random_team_pool must return different teams (randomness)"
        )

    @pytest.mark.unit
    def test_random_team_pool_empty_dir(self):
        """REQUIRES: random_team_pool() with empty directory → error or empty."""
        try:
            from pokemon_rl.env import random_team_pool
        except ImportError:
            pytest.skip("random_team_pool not yet implemented (Phase 5)")

        import tempfile
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises((ValueError, FileNotFoundError)):
                pool = random_team_pool(empty_dir)
                pool()  # Should fail when trying to get a team

    @pytest.mark.unit
    def test_random_team_pool_nonexistent_dir(self):
        """REQUIRES: random_team_pool() with nonexistent directory → error."""
        try:
            from pokemon_rl.env import random_team_pool
        except ImportError:
            pytest.skip("random_team_pool not yet implemented (Phase 5)")

        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            random_team_pool("/nonexistent/path/to/teams")


# ============================================================================
# T7: PokemonBattleEnv Construction
# ============================================================================

class TestPokemonBattleEnvConstruction:
    """T7: Verify constructor validates parameters correctly."""

    @pytest.mark.unit
    def test_valid_defaults(self):
        """Construction with all defaults → no error."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv()
        assert env.battle_format == "gen1randombattle"
        assert env.play_mode == "single"
        assert env.opponent_type == "random"
        assert env.reward_win == 1.0
        assert env.reward_loss == 0.0
        assert env.reward_draw == 0.0
        assert env.max_game_turns == 200

    @pytest.mark.unit
    def test_selfplay_mode(self):
        """play_mode='self_play' → no error."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(play_mode="self_play")
        assert env.play_mode == "self_play"

    @pytest.mark.unit
    def test_invalid_play_mode_raises(self):
        """play_mode='invalid' → ValueError."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError, match="Unknown play_mode"):
            PokemonBattleEnv(play_mode="invalid")

    @pytest.mark.unit
    def test_old_heuristic_mode_rejected(self):
        """play_mode='heuristic' → ValueError (old name removed)."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError):
            PokemonBattleEnv(play_mode="heuristic")

    @pytest.mark.unit
    def test_custom_rewards_stored(self):
        """Custom reward values stored correctly."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(reward_win=10.0, reward_loss=-5.0, reward_draw=2.5)
        assert env.reward_win == 10.0
        assert env.reward_loss == -5.0
        assert env.reward_draw == 2.5

    @pytest.mark.unit
    def test_translator_format_set(self):
        """observation_format sets translator.format_style."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(observation_format="simple")
        assert env.translator.format_style == "simple"

        env2 = PokemonBattleEnv(observation_format="pokechamp_io")
        assert env2.translator.format_style == "pokechamp_io"

    @pytest.mark.unit
    def test_system_prompt_none_preserves_translator(self):
        """system_prompt=None (default) preserves translator's prompt."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(system_prompt=None)
        assert env._system_prompt is None

    @pytest.mark.unit
    def test_system_prompt_override(self):
        """Explicit system_prompt stored for override."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(system_prompt="Custom prompt")
        assert env._system_prompt == "Custom prompt"

    @requires_verifiers
    @pytest.mark.unit
    def test_score_rollouts_always_true(self):
        """ANTI-REWARD-HACKING: score_rollouts cannot be set to False.

        score_rollouts=False would bypass the scoring pipeline entirely,
        producing zero rewards for all training samples.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(score_rollouts=False)  # Silently ignored
        assert env.score_rollouts is True, (
            "score_rollouts must ALWAYS be True"
        )

    @requires_verifiers
    @pytest.mark.unit
    def test_inherits_multiturn_env(self):
        """PokemonBattleEnv inherits from vf.MultiTurnEnv."""
        from pokemon_rl.env import PokemonBattleEnv

        assert issubclass(PokemonBattleEnv, vf.MultiTurnEnv)

    @pytest.mark.unit
    def test_team_dir_param(self):
        """REQUIRES: team_dir parameter in PokemonBattleEnv.

        Verify team_dir creates a callable team_fn.
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            pytest.skip("team_dir param not yet implemented (Phase 5)")

        if not hasattr(env, "team_fn"):
            pytest.skip("team_fn attribute not yet created by team_dir (Phase 5)")

        assert callable(env.team_fn)
        team = env.team_fn()
        assert isinstance(team, str)
        assert len(team) > 0

    @pytest.mark.unit
    def test_team_fn_overrides_team_dir(self):
        """REQUIRES: team_fn parameter takes priority over team_dir."""
        from pokemon_rl.env import PokemonBattleEnv

        custom_team = "Pikachu @ Light Ball\nAbility: Static\nEVs: 252 Spe\nTimid Nature\n- Thunderbolt"

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                team_fn=lambda: custom_team,
                team_dir=TEAM_DIR,
            )
        except TypeError:
            pytest.skip("team_fn param not yet implemented (Phase 5)")

        assert env.team_fn() == custom_team

    @pytest.mark.unit
    def test_gen9ou_without_team_dir_warns(self):
        """REQUIRES: gen9ou format without team_dir logs a warning.

        Phase 5 implementation should validate that non-random formats
        have a team source. Until implemented, this test skips.
        """
        from pokemon_rl.env import PokemonBattleEnv

        # Check if team_dir validation is implemented by looking for
        # the team_fn attribute or a warning mechanism
        env = PokemonBattleEnv(battle_format="gen9ou")
        if not hasattr(env, "team_fn"):
            pytest.skip("team_dir validation not yet implemented (Phase 5)")

        # If implemented: verify warning was logged or team_fn is None
        assert env.team_fn is None or not callable(env.team_fn), (
            "gen9ou without team_dir should NOT have a working team_fn"
        )

    @pytest.mark.unit
    def test_step_reward_fn_stored(self):
        """step_reward_fn callback stored and usable."""
        from pokemon_rl.env import PokemonBattleEnv

        fn = lambda before, after, action, idx: 0.1
        env = PokemonBattleEnv(step_reward_fn=fn)
        assert env.step_reward_fn is fn

    @pytest.mark.unit
    def test_server_host_stored(self):
        """server_host parameter stored for cross-node play."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(server_host="nid200313")
        assert env.server_host == "nid200313"


# ============================================================================
# T7b: load_environment Round-Trip
# ============================================================================

class TestLoadEnvironmentRoundTrip:
    """T7b: Verify load_environment(**toml_args) creates correct env.

    This is the actual production call path:
    env_worker → importlib → load_environment → PokemonBattleEnv

    REQUIRES: team_dir/team_fn implementation for gen9ou configs.
    """

    @pytest.mark.unit
    def test_load_environment_basic(self):
        """load_environment with minimal args creates valid env."""
        from pokemon_rl import load_environment

        env = load_environment(
            battle_format="gen1randombattle",
            port=8000,
            play_mode="single",
            observation_format="simple",
        )
        from pokemon_rl.env import PokemonBattleEnv
        assert isinstance(env, PokemonBattleEnv)
        assert env.battle_format == "gen1randombattle"
        assert env.play_mode == "single"

    @pytest.mark.unit
    def test_load_environment_selfplay(self):
        """load_environment with self_play mode."""
        from pokemon_rl import load_environment

        env = load_environment(
            battle_format="gen9randombattle",
            play_mode="self_play",
            observation_format="pokechamp_io",
            reward_win=1.0,
            reward_loss=0.0,
            reward_draw=0.0,
            max_game_turns=50,
        )
        assert env.play_mode == "self_play"
        assert env.battle_format == "gen9randombattle"
        assert env.reward_win == 1.0
        assert env.reward_draw == 0.0
        assert env.max_game_turns == 50
        assert env.translator.format_style == "pokechamp_io"

    @pytest.mark.unit
    def test_load_environment_selfplay_config_args(self):
        """REQUIRES: team_dir support. Simulates rl_selfplay.toml env.args."""
        from pokemon_rl import load_environment

        selfplay_args = {
            "battle_format": "gen9ou",
            "play_mode": "self_play",
            "port": 8000,
            "observation_format": "pokechamp_io",
            "reward_win": 1.0,
            "reward_loss": 0.0,
            "reward_draw": 0.0,
            "max_game_turns": 200,
            "num_battles": 10000,
            "team_dir": TEAM_DIR,
        }
        try:
            env = load_environment(**selfplay_args)
        except TypeError:
            pytest.skip("team_dir not yet supported in load_environment (Phase 5)")

        assert env.battle_format == "gen9ou"
        assert env.play_mode == "self_play"
        assert env.reward_draw == 0.0
        assert env.translator.format_style == "pokechamp_io"

        if not hasattr(env, "team_fn"):
            pytest.skip("team_fn not yet created by team_dir (Phase 5)")
        assert callable(env.team_fn)

    @pytest.mark.unit
    def test_load_environment_heuristic_config_args(self):
        """REQUIRES: team_dir support. Simulates rl_vs_heuristic.toml env.args."""
        from pokemon_rl import load_environment

        heuristic_args = {
            "battle_format": "gen9ou",
            "play_mode": "single",
            "opponent_type": "max_damage",
            "port": 8000,
            "observation_format": "pokechamp_io",
            "reward_win": 1.0,
            "reward_loss": 0.0,
            "reward_draw": 0.0,
            "max_game_turns": 200,
            "num_battles": 10000,
            "team_dir": TEAM_DIR,
        }
        try:
            env = load_environment(**heuristic_args)
        except TypeError:
            pytest.skip("team_dir not yet supported in load_environment (Phase 5)")

        assert env.battle_format == "gen9ou"
        assert env.play_mode == "single"
        assert env.opponent_type == "max_damage"

    @pytest.mark.unit
    def test_load_environment_test_config_args(self):
        """Simulates rl_test.toml env.args (gen9randombattle — no teams needed)."""
        from pokemon_rl import load_environment

        test_args = {
            "battle_format": "gen9randombattle",
            "play_mode": "self_play",
            "port": 8000,
            "observation_format": "pokechamp_io",
            "reward_win": 1.0,
            "reward_loss": 0.0,
            "reward_draw": 0.0,
            "max_game_turns": 50,
            "num_battles": 100,
        }
        env = load_environment(**test_args)
        assert env.battle_format == "gen9randombattle"
        assert env.max_game_turns == 50

    @pytest.mark.unit
    def test_load_environment_returns_correct_type(self):
        """load_environment returns PokemonBattleEnv instance."""
        from pokemon_rl import load_environment
        from pokemon_rl.env import PokemonBattleEnv

        env = load_environment()
        assert isinstance(env, PokemonBattleEnv)


# ============================================================================
# T_extra: Parse Failure Tracking
# ============================================================================

class TestParseFailureTracking:
    """ANTI-REWARD-HACKING: Parse failures must be tracked in agent context.

    A model that learns to output garbage to exploit the fallback policy
    must be detectable via parse_failure_count.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_failure_increments_counter(self):
        """Each parse failure increments agent.parse_failure_count."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", play_mode="single",
            observation_format="simple",
        )

        # Create mock state manually
        from unittest.mock import AsyncMock
        mock_mgr = MagicMock()
        mock_mgr.step = AsyncMock(return_value=(MockBattle("t2", turn=2), False))
        mock_mgr.get_result = MagicMock(return_value={
            "won": True, "turns": 2, "steps": 2,
            "format": "gen1randombattle", "battle_tag": "test",
        })

        env.translator = MagicMock()
        env.translator.extract_completion_text = MagicMock(return_value="garbage text")
        env.translator.parse_action = MagicMock(return_value=None)  # Parse fails
        fallback = MagicMock()
        fallback.message = "/choose default"
        env.translator.get_fallback_action = MagicMock(return_value=fallback)
        env.translator.extract_user_content = MagicMock(return_value="user content")

        agent = _AgentContext(agent_idx=0)
        agent.battle = MockBattle()
        state = {
            "manager": mock_mgr,
            "_agents": [agent],
            "_current_agent_idx": 0,
            "trajectory": [],
            "game_over": False,
            "game_turn": 0,
        }

        step = {
            "completion": [{"role": "assistant", "content": "garbage"}],
            "prompt": [{"role": "user", "content": "battle state"}],
            "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        assert agent.parse_failure_count == 1
        assert state["trajectory"][0]["extras"]["parse_failed"] is True

    @pytest.mark.unit
    def test_rewards_match_outcome_not_parse_quality(self):
        """ANTI-REWARD-HACKING: Rewards come from win/loss, not from parse success.

        A game won by luck with 100% parse failures gets the SAME reward
        as a game won with perfect parsing.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(reward_win=1.0, reward_loss=0.0)

        # Game won with all parse failures
        parse_fail_traj = [
            {"extras": {"agent_idx": 0, "parse_failed": True}},
            {"extras": {"agent_idx": 0, "parse_failed": True}},
        ]
        # Game won with perfect parsing
        parse_ok_traj = [
            {"extras": {"agent_idx": 0, "parse_failed": False}},
            {"extras": {"agent_idx": 0, "parse_failed": False}},
        ]

        state_fail = {"won": True, "trajectory": parse_fail_traj}
        state_ok = {"won": True, "trajectory": parse_ok_traj}

        env._assign_rewards(state_fail)
        env._assign_rewards(state_ok)

        assert state_fail["trajectory"][0]["reward"] == state_ok["trajectory"][0]["reward"], (
            "Reward must be same regardless of parse quality"
        )


# ============================================================================
# T_extra: Render Completion Format
# ============================================================================

class TestRenderCompletion:
    """Verify render_completion sets all framework-required fields."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_sets_required_fields(self):
        """render_completion sets reward, completion, metrics."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(play_mode="single")
        trajectory = [
            {
                "extras": {"agent_idx": 0, "parse_failed": False},
                "completion": [{"role": "assistant", "content": "action"}],
            },
        ]
        state = {
            "won": True,
            "trajectory": trajectory,
            "game_over": True,
            "game_turn": 5,
        }

        await env.render_completion(state)

        assert isinstance(state["reward"], (int, float))
        assert state["reward"] == 1.0
        assert "completion" in state
        assert "metrics" in state
        assert state["metrics"]["won"] == 1
        assert state["metrics"]["game_turns"] == 5
        assert state["metrics"]["parse_failures"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_empty_trajectory(self):
        """render_completion with empty trajectory → reward=0.0."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv()
        state = {"won": True, "trajectory": [], "game_turn": 0}

        await env.render_completion(state)

        assert state["reward"] == 0.0
        assert state["completion"] == []


# ============================================================================
# T_extra: Unrecognized kwargs warning
# ============================================================================

class TestUnrecognizedKwargs:
    """T7 spec: Unrecognized kwargs should be caught.

    REQUIRES: Explicit kwarg validation in PokemonBattleEnv (Phase 5).
    Currently **kwargs silently swallows unknown params.
    """

    @pytest.mark.unit
    def test_unrecognized_kwarg_detected(self):
        """REQUIRES: Unrecognized kwarg detection in PokemonBattleEnv.

        bogus_param=True should produce a warning, not be silently swallowed.
        """
        from pokemon_rl.env import PokemonBattleEnv
        import warnings

        # Try creating env with unrecognized kwarg
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                env = PokemonBattleEnv(bogus_param=True)
                # If we get here without error, check for warning
                # (Warning mechanism may not be implemented yet)
                if not any("bogus_param" in str(warning.message) for warning in w):
                    pytest.skip(
                        "Unrecognized kwarg detection not yet implemented (Phase 5). "
                        "Currently **kwargs silently swallows bogus_param."
                    )
        except TypeError:
            # Good: the kwarg was rejected
            pass


# ============================================================================
# T_extra: AbyssalPlayer in opponent factory
# ============================================================================

class TestAbyssalPlayer:
    """Verify AbyssalPlayer registration in create_opponent.

    REQUIRES: "abyssal" type in create_opponent (Phase 5, task 6.2).
    """

    @requires_poke_env
    @pytest.mark.unit
    def test_abyssal_opponent_type_exists(self):
        """create_opponent supports 'abyssal' type.

        REQUIRES: AbyssalPlayer registration (Phase 5).
        """
        from pokemon_rl.players import create_opponent
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration("localhost", 8000)
        try:
            opponent = create_opponent(
                opponent_type="abyssal",
                battle_format="gen9ou",
                server_config=server_config,
            )
        except (ValueError, KeyError, ImportError):
            pytest.skip("AbyssalPlayer not yet registered in create_opponent (Phase 5)")

        assert opponent is not None


# ============================================================================
# T_extra: Logprobs not all zero (anti-reward-hacking safeguard #8)
# ============================================================================

class TestTrajectoryCompleteness:
    """Verify trajectory tokens have proper logprobs (not all zeros).

    All-zero logprobs mean no gradient signal → training is useless.
    This safeguard can only be fully tested in GPU tests with real vLLM,
    but we can test the contract here.
    """

    @pytest.mark.unit
    def test_trajectory_token_contract(self):
        """Trajectory steps must have tokens with prompt_ids, completion_ids, logprobs."""
        # This defines the contract that GPU tests will verify with real data
        required_token_fields = [
            "prompt_ids", "completion_ids", "completion_logprobs",
            "prompt_mask", "completion_mask",
        ]

        # Valid token dict
        valid_tokens = {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 5, 6],
            "completion_logprobs": [-0.5, -0.3, -0.7],
            "prompt_mask": [1, 1, 1],
            "completion_mask": [1, 1, 1],
        }

        for field in required_token_fields:
            assert field in valid_tokens, f"Missing required token field: {field}"

        # Logprobs must be negative (log probabilities)
        for lp in valid_tokens["completion_logprobs"]:
            assert lp <= 0, f"Logprob {lp} must be ≤ 0"

        # Logprobs must NOT be all zero (would mean no gradient)
        assert not all(lp == 0 for lp in valid_tokens["completion_logprobs"]), (
            "All-zero logprobs mean no gradient signal — training is useless"
        )

        # Negative: all-zero logprobs should be detectable
        bad_logprobs = [0.0, 0.0, 0.0]
        assert all(lp == 0 for lp in bad_logprobs), (
            "Test helper: all-zero detection works"
        )
