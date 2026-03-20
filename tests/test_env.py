"""Tests for Layer 4: PokemonBattleEnv.

Unit tests validate the env state machine with mocks.
Integration tests run full game loops with real Showdown.
"""

import pytest

from tests.conftest import requires_poke_env, requires_showdown


# ---- Unit tests: env state machine ----


class TestEnvStateMachine:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_setup_state(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            adapter=None, translator=None, opponent_mode="heuristic"
        )
        state = await env.setup_state({})

        assert state["trajectory"] == []
        assert state["game_over"] is False
        assert state["turn"] == 0
        assert state["winner"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_game_over(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {"game_over": True, "turn": 10}

        result = await env.get_prompt_messages(state)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_max_turns(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None, max_game_turns=5)
        state = {"game_over": False, "turn": 5, "battle": None}

        result = await env.get_prompt_messages(state)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_trajectory_step(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {"trajectory": [], "turn": 0, "battle": None}

        step = {"completion": '{"move": "thunderbolt"}'}
        await env.add_trajectory_step(state, step)

        assert len(state["trajectory"]) == 1
        assert state["turn"] == 1
        assert state["trajectory"][0]["player_idx"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_win(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": True,
            "turn": 15,
            "trajectory": [
                {"player_idx": 0},
                {"player_idx": 0},
                {"player_idx": 0},
            ],
        }

        await env.render_completion(state)

        assert state["reward"] == 1.0
        assert all(s["reward"] == 1.0 for s in state["trajectory"])
        assert state["metrics"]["won"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_loss(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": False,
            "turn": 10,
            "trajectory": [{"player_idx": 0}, {"player_idx": 0}],
        }

        await env.render_completion(state)

        assert state["reward"] == 0.0
        assert all(s["reward"] == 0.0 for s in state["trajectory"])
        assert state["metrics"]["won"] == 0

    @pytest.mark.unit
    def test_self_play_not_implemented(self):
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(NotImplementedError):
            PokemonBattleEnv(adapter=None, translator=None, opponent_mode="self_play")

    @pytest.mark.unit
    def test_passthrough_reward(self):
        from pokemon_rl.env import _passthrough_reward

        assert _passthrough_reward({"reward": 1.0}) == 1.0
        assert _passthrough_reward({"reward": 0.0}) == 0.0
        assert _passthrough_reward({}) == 0.0


# ---- Integration tests: full game loop ----


@requires_poke_env
@requires_showdown
class TestEnvIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_standalone_simple(self, showdown_port):
        """Full game loop: env + adapter + simple translator."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(adapter=adapter, translator=translator)

        result = await env.run_standalone()

        assert result["turns"] > 0
        assert len(result["trajectory"]) > 0
        assert result["won"] in (True, False, None)
        assert result["reward"] in (0.0, 1.0)

        # All trajectory steps should have rewards
        for step in result["trajectory"]:
            assert "reward" in step
            assert step["reward"] == result["reward"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_standalone_with_random_action(self, showdown_port):
        """Full game with random moves."""
        from pokemon_rl.adapter import BattleAdapter, random_action
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(adapter=adapter, translator=translator)

        result = await env.run_standalone(action_fn=random_action)

        assert result["turns"] > 0
        assert "battle_tag" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trajectory_prompt_lengths(self, showdown_port):
        """Verify prompts are being generated and have reasonable length."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(adapter=adapter, translator=translator)

        result = await env.run_standalone()

        for step in result["trajectory"]:
            assert step["prompt_length"] > 0
            assert step["prompt_messages"] == 2  # system + user
