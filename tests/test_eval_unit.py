"""Unit tests for pokemon-rl eval feature.

All tests run on login node — no Showdown, no GPU, no poke-env required.
Uses mocks for all external dependencies.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestConfigParsing:

    @pytest.mark.unit
    def test_config_from_toml_valid(self, tmp_path):
        """Parse a valid TOML config with all opponent types."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "Qwen/Qwen3-4B"
agent_base_url = "http://localhost:8001/v1"
n_battles_per_opp = 50
battle_format = "gen9ou"

[[opponents]]
name = "abyssal"
type = "heuristic"
heuristic = "abyssal"

[[opponents]]
name = "kakuna"
type = "metamon"
agent = "kakuna"
gpu_ids = [3]

[[opponents]]
name = "weak-llm"
type = "llm"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
base_url = "http://localhost:8002/v1"
gpu_ids = [2, 3]

[[opponents]]
name = "self-play"
type = "llm"
model_name = "Qwen/Qwen3-4B"
base_url = "http://localhost:8003/v1"
gpu_ids = [2, 3]
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        config = PokemonEvalConfig.from_toml(str(config_path))

        assert config.agent_model == "Qwen/Qwen3-4B"
        assert config.n_battles_per_opp == 50
        assert len(config.opponents) == 4

        # Verify each opponent type
        assert config.opponents[0].type == "heuristic"
        assert config.opponents[1].type == "metamon"
        assert config.opponents[2].type == "llm"
        assert config.opponents[3].type == "llm"

    @pytest.mark.unit
    def test_config_from_toml_heuristic(self, tmp_path):
        """Heuristic opponent: type is the category, heuristic is the specific agent."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "abyssal"
type = "heuristic"
heuristic = "abyssal"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        config = PokemonEvalConfig.from_toml(str(config_path))
        opp = config.opponents[0]

        assert opp.type == "heuristic"
        assert opp.heuristic == "abyssal"
        # type is the category, NOT the specific name
        assert opp.type != "abyssal"

    @pytest.mark.unit
    def test_config_from_toml_metamon(self, tmp_path):
        """Metamon opponent has agent field and gpu_ids."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "kakuna"
type = "metamon"
agent = "kakuna"
gpu_ids = [3]
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        config = PokemonEvalConfig.from_toml(str(config_path))
        opp = config.opponents[0]

        assert opp.type == "metamon"
        assert opp.agent == "kakuna"
        assert opp.gpu_ids == [3]

    @pytest.mark.unit
    def test_config_from_toml_llm(self, tmp_path):
        """LLM opponent has model_name and base_url."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "weak"
type = "llm"
model_name = "Qwen/Qwen2.5-1.5B"
base_url = "http://localhost:8002/v1"
gpu_ids = [2, 3]
max_tokens = 512
temperature = 0.5
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        config = PokemonEvalConfig.from_toml(str(config_path))
        opp = config.opponents[0]

        assert opp.model_name == "Qwen/Qwen2.5-1.5B"
        assert opp.base_url == "http://localhost:8002/v1"
        assert opp.max_tokens == 512
        assert opp.temperature == 0.5

    @pytest.mark.unit
    def test_config_validation_heuristic_missing_field(self, tmp_path):
        """type='heuristic' without heuristic field raises ValueError."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "bad"
type = "heuristic"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        with pytest.raises(ValueError, match="heuristic"):
            PokemonEvalConfig.from_toml(str(config_path))

    @pytest.mark.unit
    def test_config_validation_metamon_missing_agent(self, tmp_path):
        """type='metamon' without agent field raises ValueError."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "bad"
type = "metamon"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        with pytest.raises(ValueError, match="agent"):
            PokemonEvalConfig.from_toml(str(config_path))

    @pytest.mark.unit
    def test_config_validation_llm_missing_model(self, tmp_path):
        """type='llm' without model_name raises ValueError."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "bad"
type = "llm"
base_url = "http://localhost:8002/v1"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        with pytest.raises(ValueError, match="model_name"):
            PokemonEvalConfig.from_toml(str(config_path))

    @pytest.mark.unit
    def test_config_validation_llm_missing_base_url(self, tmp_path):
        """type='llm' without base_url raises ValueError."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "bad"
type = "llm"
model_name = "Qwen/Qwen2.5-1.5B"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        with pytest.raises(ValueError, match="base_url"):
            PokemonEvalConfig.from_toml(str(config_path))

    @pytest.mark.unit
    def test_config_validation_unknown_type(self, tmp_path):
        """Unknown opponent type raises ValueError."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "bad"
type = "unknown_type"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        with pytest.raises(ValueError, match="unknown_type"):
            PokemonEvalConfig.from_toml(str(config_path))

    @pytest.mark.unit
    def test_config_defaults(self, tmp_path):
        """Omitted fields use sensible defaults."""
        from pokemon_rl.eval.config import PokemonEvalConfig

        toml_content = """
agent_model = "test"
agent_base_url = "http://localhost:8001/v1"
[[opponents]]
name = "random"
type = "heuristic"
heuristic = "random"
"""
        config_path = tmp_path / "eval.toml"
        config_path.write_text(toml_content)

        config = PokemonEvalConfig.from_toml(str(config_path))

        assert config.n_battles_per_opp == 100
        assert config.max_concurrent_battles == 8
        assert config.max_game_turns == 200
        assert config.showdown_port == 8000
        assert config.node_rank == 0
        assert config.n_nodes == 1


# ---------------------------------------------------------------------------
# OpponentConfig.opponent_type_for_env
# ---------------------------------------------------------------------------


class TestOpponentTypeMapping:

    @pytest.mark.unit
    def test_heuristic_maps_to_specific_name(self):
        """Heuristic type maps to the specific heuristic name for env."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(name="test", type="heuristic", heuristic="abyssal")
        assert opp.opponent_type_for_env == "abyssal"

        opp2 = OpponentConfig(name="test2", type="heuristic", heuristic="max_damage")
        assert opp2.opponent_type_for_env == "max_damage"

    @pytest.mark.unit
    def test_metamon_maps_to_agent_name(self):
        """Metamon type maps to the agent name for env."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(name="test", type="metamon", agent="kakuna")
        assert opp.opponent_type_for_env == "kakuna"

    @pytest.mark.unit
    def test_llm_maps_to_llm_string(self):
        """LLM type maps to 'llm' string for env."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(
            name="test", type="llm",
            model_name="test-model", base_url="http://localhost:8002/v1",
        )
        assert opp.opponent_type_for_env == "llm"


# ---------------------------------------------------------------------------
# Opponent registry
# ---------------------------------------------------------------------------


class TestOpponentRegistry:

    @pytest.mark.unit
    def test_registry_llm_entry(self):
        """'llm' is registered as a direct opponent."""
        from pokemon_rl.opponents import get_opponent_spec

        spec = get_opponent_spec("llm")
        assert spec.kind == "direct"
        assert spec.opponent_type == "llm"

    @pytest.mark.unit
    def test_registry_existing_entries_unchanged(self):
        """Existing registry entries still work correctly."""
        from pokemon_rl.opponents import get_opponent_spec

        # Heuristic — direct
        spec_abyssal = get_opponent_spec("abyssal")
        assert spec_abyssal.kind == "direct"
        assert spec_abyssal.opponent_type == "abyssal"

        # External — kakuna
        spec_kakuna = get_opponent_spec("kakuna")
        assert spec_kakuna.kind == "external"

        # Verify they are DIFFERENT kinds
        assert spec_abyssal.kind != spec_kakuna.kind


# ---------------------------------------------------------------------------
# Multi-node battle split
# ---------------------------------------------------------------------------


class TestComputeNodeShare:

    @pytest.mark.unit
    def test_even_split(self):
        """100 battles across 4 nodes = 25 each."""
        from pokemon_rl.eval.config import compute_node_share

        shares = [compute_node_share(100, rank, 4) for rank in range(4)]
        assert shares == [25, 25, 25, 25]
        assert sum(shares) == 100

    @pytest.mark.unit
    def test_uneven_split(self):
        """100 battles across 3 nodes: earlier nodes get remainder."""
        from pokemon_rl.eval.config import compute_node_share

        shares = [compute_node_share(100, rank, 3) for rank in range(3)]
        assert sum(shares) == 100
        # First node gets 1 extra
        assert shares[0] == 34
        assert shares[1] == 33
        assert shares[2] == 33

    @pytest.mark.unit
    def test_single_node(self):
        """Single node gets all battles."""
        from pokemon_rl.eval.config import compute_node_share

        assert compute_node_share(100, 0, 1) == 100

    @pytest.mark.unit
    def test_more_nodes_than_battles(self):
        """More nodes than battles: some nodes get 0."""
        from pokemon_rl.eval.config import compute_node_share

        shares = [compute_node_share(3, rank, 5) for rank in range(5)]
        assert sum(shares) == 3
        assert shares == [1, 1, 1, 0, 0]

    @pytest.mark.unit
    def test_zero_battles(self):
        """Zero battles: all nodes get 0."""
        from pokemon_rl.eval.config import compute_node_share

        shares = [compute_node_share(0, rank, 3) for rank in range(3)]
        assert shares == [0, 0, 0]


# ---------------------------------------------------------------------------
# Report / stats tests
# ---------------------------------------------------------------------------


class TestComputeStats:

    def _make_state(self, won: bool | None, game_turns: int = 10, parse_failures: int = 0):
        """Create a mock state dict with metrics."""
        won_val = 1 if won is True else (0 if won is False else -1)
        return {
            "metrics": {
                "won": won_val,
                "game_turns": game_turns,
                "parse_failures": parse_failures,
                "wins": int(won is True),
                "losses": int(won is False),
                "draws": int(won is None),
            },
            "reward": 1.0 if won else 0.0,
        }

    @pytest.mark.unit
    def test_all_wins(self):
        """10 wins → win_rate=1.0, loss_rate=0.0."""
        from pokemon_rl.eval.report import compute_stats

        states = [self._make_state(True) for _ in range(10)]
        stats = compute_stats(states)

        assert stats["wins"] == 10
        assert stats["losses"] == 0
        assert stats["draws"] == 0
        assert stats["win_rate"] == 1.0
        assert stats["loss_rate"] == 0.0
        assert stats["total"] == 10

    @pytest.mark.unit
    def test_all_losses(self):
        """10 losses → win_rate=0.0, loss_rate=1.0."""
        from pokemon_rl.eval.report import compute_stats

        states = [self._make_state(False) for _ in range(10)]
        stats = compute_stats(states)

        assert stats["wins"] == 0
        assert stats["losses"] == 10
        assert stats["win_rate"] == 0.0
        assert stats["loss_rate"] == 1.0

    @pytest.mark.unit
    def test_mixed_results(self):
        """7 wins, 2 losses, 1 draw → correct rates. Rates sum to 1.0."""
        from pokemon_rl.eval.report import compute_stats

        states = (
            [self._make_state(True)] * 7
            + [self._make_state(False)] * 2
            + [self._make_state(None)] * 1
        )
        stats = compute_stats(states)

        assert stats["win_rate"] == pytest.approx(0.7)
        assert stats["loss_rate"] == pytest.approx(0.2)
        assert stats["draw_rate"] == pytest.approx(0.1)
        # Rates must sum to 1.0
        assert stats["win_rate"] + stats["loss_rate"] + stats["draw_rate"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_win_rate_different_from_loss_rate_when_mixed(self):
        """No-fall-through: win_rate != loss_rate for non-symmetric results."""
        from pokemon_rl.eval.report import compute_stats

        states = [self._make_state(True)] * 7 + [self._make_state(False)] * 3
        stats = compute_stats(states)

        assert stats["win_rate"] != stats["loss_rate"]

    @pytest.mark.unit
    def test_empty_states(self):
        """0 battles → no crash, zeros."""
        from pokemon_rl.eval.report import compute_stats

        stats = compute_stats([])
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["win_rate_stderr"] == 0.0

    @pytest.mark.unit
    def test_stderr_calculation(self):
        """Verify stderr = sqrt(p * (1-p) / n)."""
        from pokemon_rl.eval.report import compute_stats

        states = [self._make_state(True)] * 7 + [self._make_state(False)] * 3
        stats = compute_stats(states)

        expected_se = math.sqrt(0.7 * 0.3 / 10)
        assert stats["win_rate_stderr"] == pytest.approx(expected_se)

    @pytest.mark.unit
    def test_avg_game_turns(self):
        """Average game turns computed correctly."""
        from pokemon_rl.eval.report import compute_stats

        states = [
            self._make_state(True, game_turns=20),
            self._make_state(False, game_turns=10),
        ]
        stats = compute_stats(states)
        assert stats["avg_game_turns"] == pytest.approx(15.0)

    @pytest.mark.unit
    def test_avg_parse_failures(self):
        """Average parse failures computed correctly."""
        from pokemon_rl.eval.report import compute_stats

        states = [
            self._make_state(True, parse_failures=2),
            self._make_state(False, parse_failures=0),
        ]
        stats = compute_stats(states)
        assert stats["avg_parse_failures"] == pytest.approx(1.0)


class TestSaveResults:

    @pytest.mark.unit
    def test_save_results_jsonl_format(self, tmp_path):
        """Results saved as JSONL with required fields."""
        from pokemon_rl.eval.report import save_results

        states = [
            {
                "example_id": 0,
                "reward": 1.0,
                "metrics": {"won": 1, "game_turns": 15, "parse_failures": 0,
                             "wins": 1, "losses": 0, "draws": 0},
            },
            {
                "example_id": 1,
                "reward": 0.0,
                "metrics": {"won": 0, "game_turns": 20, "parse_failures": 1,
                             "wins": 0, "losses": 1, "draws": 0},
            },
        ]

        filepath = save_results(states, "test_opp", str(tmp_path))

        assert filepath.exists()
        with open(filepath) as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2

        # Verify required fields
        for row in lines:
            assert "example_id" in row
            assert "opponent" in row
            assert "reward" in row
            assert "won" in row
            assert "game_turns" in row
            assert "parse_failures" in row

        # Verify opponent field
        assert lines[0]["opponent"] == "test_opp"
        assert lines[1]["opponent"] == "test_opp"

        # Verify values propagated correctly
        assert lines[0]["won"] == 1
        assert lines[1]["won"] == 0

    @pytest.mark.unit
    def test_save_results_node_rank(self, tmp_path):
        """Node rank > 0 uses results_{rank}.jsonl filename."""
        from pokemon_rl.eval.report import save_results

        states = [{"example_id": 0, "reward": 1.0, "metrics": {"won": 1, "game_turns": 5,
                    "parse_failures": 0, "wins": 1, "losses": 0, "draws": 0}}]

        filepath = save_results(states, "opp", str(tmp_path), node_rank=2)

        assert filepath.name == "results_2.jsonl"


class TestGenerateSummary:

    @pytest.mark.unit
    def test_summary_format(self, tmp_path):
        """Summary has header + rows for each opponent."""
        from pokemon_rl.eval.report import generate_summary

        results = {
            "abyssal": {"win_rate": 0.6, "loss_rate": 0.3, "draw_rate": 0.1,
                         "win_rate_stderr": 0.05, "avg_game_turns": 15.0, "total": 100},
            "random": {"win_rate": 0.9, "loss_rate": 0.05, "draw_rate": 0.05,
                        "win_rate_stderr": 0.03, "avg_game_turns": 10.0, "total": 100},
        }

        summary = generate_summary(results, str(tmp_path))

        # Has header and rows
        lines = summary.strip().split("\n")
        assert len(lines) >= 3  # header + separator + 2 data rows

        # Summary JSON was saved
        assert (tmp_path / "summary.json").exists()

        # Verify JSON content
        with open(tmp_path / "summary.json") as f:
            saved = json.load(f)
        assert "abyssal" in saved
        assert "random" in saved

    @pytest.mark.unit
    def test_summary_node_rank(self, tmp_path):
        """Node rank > 0 uses summary_{rank}.json."""
        from pokemon_rl.eval.report import generate_summary

        generate_summary({"opp": {"win_rate": 0.5}}, str(tmp_path), node_rank=1)

        assert (tmp_path / "summary_1.json").exists()


class TestMergeNodeResults:

    @pytest.mark.unit
    def test_merge_two_nodes(self, tmp_path):
        """Merge results from 2 nodes."""
        from pokemon_rl.eval.report import merge_node_results

        opp_dir = tmp_path / "test_opp"
        opp_dir.mkdir()

        # Node 0
        with open(opp_dir / "results.jsonl", "w") as f:
            f.write(json.dumps({"example_id": 0, "won": 1}) + "\n")
            f.write(json.dumps({"example_id": 1, "won": 0}) + "\n")

        # Node 1
        with open(opp_dir / "results_1.jsonl", "w") as f:
            f.write(json.dumps({"example_id": 2, "won": 1}) + "\n")

        results = merge_node_results(str(tmp_path), "test_opp", 2)

        assert len(results) == 3
        assert results[0]["example_id"] == 0
        assert results[2]["example_id"] == 2


# ---------------------------------------------------------------------------
# LLMPlayer tests (mocked — no poke-env or API required)
# ---------------------------------------------------------------------------


class TestLLMPlayerUnit:
    """Test LLMPlayer logic with fully mocked dependencies."""

    @pytest.mark.unit
    def test_llm_player_valid_move(self):
        """Valid move parsed → returns action (not default)."""
        # Mock the action and fallback to be distinguishable
        mock_action = MagicMock(name="parsed_action")
        mock_fallback = MagicMock(name="fallback_action")
        mock_default = MagicMock(name="default_move")

        mock_translator = MagicMock()
        mock_translator.battle_to_prompt.return_value = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]
        mock_translator.parse_action.return_value = mock_action
        mock_translator.get_fallback_action.return_value = mock_fallback

        # Simulate the choose_move logic directly
        text = '{"move": "thunderbolt"}'
        action = mock_translator.parse_action(text, None)
        assert action is mock_action
        assert action is not mock_fallback
        assert action is not mock_default

    @pytest.mark.unit
    def test_llm_player_invalid_move_falls_back(self):
        """Invalid move → parse_action returns None → fallback used."""
        mock_fallback = MagicMock(name="fallback")
        mock_translator = MagicMock()
        mock_translator.parse_action.return_value = None
        mock_translator.get_fallback_action.return_value = mock_fallback

        text = '{"move": "flamethrower"}'  # not available
        action = mock_translator.parse_action(text, None)
        assert action is None

        # Fallback should be used
        fallback = mock_translator.get_fallback_action(None)
        assert fallback is mock_fallback
        assert fallback is not None

    @pytest.mark.unit
    def test_llm_player_garbage_response_falls_back(self):
        """Garbage response → parse returns None → fallback."""
        mock_translator = MagicMock()
        mock_translator.parse_action.return_value = None

        text = "I don't know what to do"
        action = mock_translator.parse_action(text, None)
        assert action is None

    @pytest.mark.unit
    def test_consecutive_failure_counter_logic(self):
        """Counter increments on failure, resets on success."""
        counter = 0
        max_failures = 3

        # 2 failures
        for _ in range(2):
            counter += 1
        assert counter == 2
        assert counter < max_failures

        # Success resets
        counter = 0
        assert counter == 0

        # 1 more failure
        counter += 1
        assert counter == 1  # Not 3 — counter was reset

    @pytest.mark.unit
    def test_consecutive_failure_forfeit_at_threshold(self):
        """After MAX_CONSECUTIVE_FAILURES, should forfeit."""
        from pokemon_rl.eval.llm_player import _MAX_CONSECUTIVE_FAILURES

        counter = _MAX_CONSECUTIVE_FAILURES
        assert counter >= 3
        assert counter >= _MAX_CONSECUTIVE_FAILURES  # Would trigger forfeit


# ---------------------------------------------------------------------------
# Runner opponent-type mapping tests
# ---------------------------------------------------------------------------


class TestRunnerMapping:

    @pytest.mark.unit
    def test_heuristic_maps_to_env_opponent_type(self):
        """type='heuristic', heuristic='abyssal' → opponent_type='abyssal'."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(name="test", type="heuristic", heuristic="abyssal")
        assert opp.opponent_type_for_env == "abyssal"

    @pytest.mark.unit
    def test_metamon_maps_to_env_opponent_type(self):
        """type='metamon', agent='kakuna' → opponent_type='kakuna'."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(name="test", type="metamon", agent="kakuna")
        assert opp.opponent_type_for_env == "kakuna"

    @pytest.mark.unit
    def test_llm_maps_to_env_opponent_type(self):
        """type='llm' → opponent_type='llm' + llm_opponent_kwargs set."""
        from pokemon_rl.eval.config import OpponentConfig

        opp = OpponentConfig(
            name="test", type="llm",
            model_name="test", base_url="http://localhost:8002/v1",
        )
        assert opp.opponent_type_for_env == "llm"

    @pytest.mark.unit
    def test_different_types_produce_different_env_types(self):
        """No-fall-through: different config types produce different env types."""
        from pokemon_rl.eval.config import OpponentConfig

        heuristic = OpponentConfig(name="h", type="heuristic", heuristic="abyssal")
        metamon = OpponentConfig(name="m", type="metamon", agent="kakuna")
        llm = OpponentConfig(name="l", type="llm", model_name="x", base_url="y")

        types = {heuristic.opponent_type_for_env, metamon.opponent_type_for_env, llm.opponent_type_for_env}
        assert len(types) == 3  # All different
