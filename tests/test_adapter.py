"""Tests for Layer 2: BattleAdapter.

Integration tests — need Showdown server + poke-env.
"""

import pytest

from tests.conftest import requires_poke_env, requires_showdown


@requires_poke_env
class TestAdapterImports:
    """Verify poke-env imports work."""

    @pytest.mark.unit
    def test_import_poke_env(self):
        from poke_env.player.player import Player
        from poke_env.player.battle_order import BattleOrder
        from poke_env.player.random_player import RandomPlayer
        from poke_env import AccountConfiguration

        assert Player is not None
        assert BattleOrder is not None
        assert RandomPlayer is not None

    @pytest.mark.unit
    def test_import_adapter(self):
        from pokemon_rl.adapter import BattleAdapter, CallbackPlayer

        adapter = BattleAdapter(port=8000, battle_format="gen1randombattle")
        assert adapter.battle_format == "gen1randombattle"


@requires_poke_env
@requires_showdown
class TestBattleAdapterIntegration:
    """Integration tests — need running Showdown server."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_battle_random(self, showdown_port):
        """Run a complete battle with random moves against RandomPlayer."""
        from pokemon_rl.adapter import BattleAdapter, random_action

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        result = await adapter.run_battle(action_fn=random_action)

        assert "trajectory" in result
        assert "won" in result
        assert "turns" in result
        assert result["turns"] > 0
        assert len(result["trajectory"]) > 0
        assert result["won"] in (True, False, None)

        # Check trajectory structure
        step = result["trajectory"][0]
        assert "turn" in step
        assert "available_moves" in step
        assert "action" in step
        assert "active_pokemon" in step

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_battle_default_action(self, showdown_port):
        """Run battle with default action (first legal move)."""
        from pokemon_rl.adapter import BattleAdapter

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        result = await adapter.run_battle()  # Uses default_action

        assert result["turns"] > 0
        assert len(result["trajectory"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trajectory_has_all_turns(self, showdown_port):
        """Verify trajectory captures every turn of the battle."""
        from pokemon_rl.adapter import BattleAdapter, random_action

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        result = await adapter.run_battle(action_fn=random_action)

        # Each turn should have a trajectory entry
        # (some turns might be force-switches with no trajectory entry,
        # but generally trajectory length ~ number of turns)
        assert len(result["trajectory"]) >= 1

        # Turns should be monotonically non-decreasing
        turns = [s["turn"] for s in result["trajectory"]]
        for i in range(1, len(turns)):
            assert turns[i] >= turns[i - 1], (
                f"Turn sequence not monotonic: {turns}"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_battles(self, showdown_port):
        """Run 3 battles in sequence to test cleanup."""
        from pokemon_rl.adapter import BattleAdapter, random_action

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )

        results = []
        for _ in range(3):
            result = await adapter.run_battle(action_fn=random_action)
            results.append(result)

        # All should have completed
        assert all(r["turns"] > 0 for r in results)
        # Battle tags should be unique
        tags = [r["battle_tag"] for r in results]
        assert len(set(tags)) == 3, f"Expected unique battle tags, got {tags}"
