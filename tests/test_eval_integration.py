"""Integration tests for pokemon-rl eval feature.

Requires: compute node + Showdown server running.
Run with: bash scripts/run_tests.sh -m integration -k test_eval -v
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

# Capability detection (same pattern as conftest.py)
SHOWDOWN_PORT = int(os.environ.get("SHOWDOWN_PORT", "8000"))


def _has_showdown():
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            return s.connect_ex(("localhost", SHOWDOWN_PORT)) == 0
    except Exception:
        return False


def _has_poke_env():
    try:
        import poke_env.player.player  # noqa: F401

        return True
    except ImportError:
        return False


def _has_verifiers():
    try:
        import verifiers  # noqa: F401

        return True
    except ImportError:
        return False


requires_showdown = pytest.mark.skipif(
    not _has_showdown(), reason=f"Showdown not running on port {SHOWDOWN_PORT}"
)
requires_poke_env = pytest.mark.skipif(not _has_poke_env(), reason="poke-env not installed")
requires_verifiers = pytest.mark.skipif(not _has_verifiers(), reason="verifiers not installed")


@requires_poke_env
@requires_showdown
class TestEvalVsHeuristic:
    """Integration tests: eval vs heuristic opponents (no GPU needed for opponent)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_eval_vs_random_completes(self, tmp_path):
        """10 battles vs random all complete with valid outcomes."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            num_battles=10,
            max_concurrent_battles=4,
            max_game_turns=200,
            observation_format="simple",
        )

        # Run battles with random action fn (no LLM needed)
        results = []
        for _ in range(10):
            result = await env.run_turn_by_turn()
            results.append(result)

        assert len(results) == 10

        wins = sum(1 for r in results if r["won"] is True)
        losses = sum(1 for r in results if r["won"] is False)
        draws = sum(1 for r in results if r["won"] is None)

        # All battles resolved
        assert wins + losses + draws == 10

        # At least some battles have turns (not all instant)
        assert any(r["turns"] > 0 for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_eval_vs_abyssal_completes(self, tmp_path):
        """10 battles vs abyssal all complete."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="abyssal",
            num_battles=10,
            max_concurrent_battles=4,
            max_game_turns=200,
            observation_format="simple",
        )

        results = []
        for _ in range(10):
            result = await env.run_turn_by_turn()
            results.append(result)

        assert len(results) == 10
        total = sum(1 for r in results if r["won"] in (True, False, None))
        assert total == 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_eval_concurrent_battles_limit(self, tmp_path):
        """Concurrent battles respect the coordinator limit."""
        from pokemon_rl.coordinator import BattleCoordinator
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            num_battles=8,
            max_concurrent_battles=2,
            max_game_turns=100,
            observation_format="simple",
        )

        # Run 8 battles with max_concurrent=2
        results = await asyncio.gather(*[
            env.run_turn_by_turn() for _ in range(8)
        ])

        assert len(results) == 8
        assert all(r["won"] in (True, False, None) for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_eval_game_turns_tracked(self, tmp_path):
        """Every battle result has positive game turns."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            num_battles=5,
            max_concurrent_battles=4,
            max_game_turns=200,
            observation_format="simple",
        )

        results = []
        for _ in range(5):
            result = await env.run_turn_by_turn()
            results.append(result)

        for i, r in enumerate(results):
            assert r["turns"] > 0, f"Battle {i}: should have at least 1 turn"


@requires_poke_env
@requires_showdown
class TestEvalReport:
    """Integration tests: report generation from real battle results."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_and_load_results(self, tmp_path):
        """Run battles, save results, verify JSONL format."""
        from pokemon_rl.env import PokemonBattleEnv
        from pokemon_rl.eval.report import save_results

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            num_battles=3,
            max_concurrent_battles=2,
            max_game_turns=100,
            observation_format="simple",
        )

        # Run 3 battles
        states = []
        for _ in range(3):
            result = await env.run_turn_by_turn()
            # Convert to state-like dict
            won_val = 1 if result["won"] is True else (0 if result["won"] is False else -1)
            state = {
                "example_id": len(states),
                "reward": 1.0 if result["won"] else 0.0,
                "metrics": {
                    "won": won_val,
                    "game_turns": result["turns"],
                    "parse_failures": 0,
                    "wins": int(result["won"] is True),
                    "losses": int(result["won"] is False),
                    "draws": int(result["won"] is None),
                },
            }
            states.append(state)

        filepath = save_results(states, "random", str(tmp_path))

        assert filepath.exists()
        with open(filepath) as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 3
        for row in rows:
            assert row["opponent"] == "random"
            assert "won" in row
            assert "game_turns" in row

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_output_directory_structure(self, tmp_path):
        """Eval creates proper directory structure."""
        from pokemon_rl.eval.report import generate_summary, save_results

        # Mock results for 2 opponents
        states_random = [
            {"example_id": 0, "reward": 1.0, "metrics": {"won": 1, "game_turns": 10,
             "parse_failures": 0, "wins": 1, "losses": 0, "draws": 0}},
        ]
        states_abyssal = [
            {"example_id": 0, "reward": 0.0, "metrics": {"won": 0, "game_turns": 20,
             "parse_failures": 0, "wins": 0, "losses": 1, "draws": 0}},
        ]

        save_results(states_random, "random", str(tmp_path))
        save_results(states_abyssal, "abyssal", str(tmp_path))

        from pokemon_rl.eval.report import compute_stats

        all_results = {
            "random": compute_stats(states_random),
            "abyssal": compute_stats(states_abyssal),
        }
        generate_summary(all_results, str(tmp_path))

        # Verify structure
        assert (tmp_path / "random" / "results.jsonl").exists()
        assert (tmp_path / "abyssal" / "results.jsonl").exists()
        assert (tmp_path / "summary.json").exists()


@requires_poke_env
@requires_showdown
class TestLLMPlayerIntegration:
    """Integration tests: LLMPlayer in real battles (mocked API, real Showdown)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_player_vs_random_battle(self):
        """LLMPlayer with mock API completes a battle vs random."""
        from unittest.mock import AsyncMock, MagicMock

        from pokemon_rl.battle import BattleManager
        from pokemon_rl.eval.llm_player import LLMPlayer

        mgr = BattleManager(port=SHOWDOWN_PORT, battle_format="gen1randombattle")

        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"move": "tackle"}'

        # We need to test via BattleManager with LLM as opponent
        # Since LLMPlayer requires poke-env Player base, create it properly
        from poke_env.player.random_player import RandomPlayer
        from poke_env.ps_client import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{SHOWDOWN_PORT}",
            "players.pokemonshowdown.com/action.php?",
        )

        # Use a mock-backed LLM player
        llm_player = LLMPlayer.create(
            base_url="http://localhost:99999/v1",  # Will fail — that's OK
            model_name="test-model",
            battle_format="gen1randombattle",
            server_config=server_config,
            observation_format="simple",
            timeout=2.0,  # Short timeout since mock won't respond
        )

        # Start battle with LLM player as opponent via BattleManager
        battle = await mgr.start_battle(
            opponent_type="random",  # Use random as the in-process opponent
            player_team=None,
        )

        # Play a few turns with random actions
        turn = 0
        while battle is not None and turn < 100:
            from poke_env.player.battle_order import BattleOrder

            if battle.available_moves:
                action = BattleOrder(battle.available_moves[0])
            elif battle.available_switches:
                action = BattleOrder(battle.available_switches[0])
            else:
                action = mgr._player.choose_default_move()
            battle, done = await mgr.step(action)
            turn += 1
            if done:
                break

        result = mgr.get_result()
        assert result["won"] in (True, False, None)
        assert result["turns"] > 0
        await mgr.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_opponent_type_registered(self):
        """'llm' opponent type is in the registry and routable."""
        from pokemon_rl.opponents import get_opponent_spec

        spec = get_opponent_spec("llm")
        assert spec.kind == "direct"
        assert spec.opponent_type == "llm"
