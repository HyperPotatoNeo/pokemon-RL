"""Tests for Layer 4: PokemonBattleEnv.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    1. REWARD CORRECTNESS: Test that wins get 1.0, losses get 0.0, AND that
       they are NOT the same value (distinguishable outcomes).
    2. TRAJECTORY INTEGRITY: Verify step counts match, agent indices are
       correct, force_switch flags are present, game_turn numbers are recorded.
    3. SELF-PLAY SYMMETRY: Both players get different states. Rewards are
       opposite. Agent indices alternate correctly.
    4. MODE GUARDS: Invalid mode combinations raise immediately, not silently.

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
        from unittest.mock import AsyncMock, MagicMock, patch
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        mock_manager = MagicMock()
        mock_battle = MagicMock()
        mock_manager.start_battle = AsyncMock(return_value=mock_battle)
        mock_manager.close = AsyncMock()

        with patch("pokemon_rl.battle.BattleManager", return_value=mock_manager):
            state = await env.setup_state({})

        assert state["trajectory"] == []
        assert state["game_over"] is False
        assert state["game_turn"] == 0
        assert state["won"] is None
        assert state["truncated"] is False
        assert state["_agents"][0].battle is mock_battle
        assert state["_current_agent_idx"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_game_over_returns_none(self):
        """Game over → game_over stop condition returns True."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {"game_over": True, "game_turn": 10}

        # game_over hook returns True for game_over=True states
        result = await env.game_over(state)
        assert result is True, "game_over=True must stop the game"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_max_turns_triggers_game_over(self):
        """Reaching max_game_turns must set game_over, truncated via game_over hook."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            max_game_turns=5,
        )
        state = {"game_over": False, "game_turn": 5, "won": None}

        result = await env.game_over(state)
        assert result is True
        assert state["game_over"] is True, "Should set game_over flag"
        assert state["truncated"] is True, "Should set truncated flag (C2 fix)"
        assert state["won"] is None, "Truncation should not set won=False"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_below_max_turns_does_not_end(self):
        """Below max_game_turns should NOT set game_over."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            max_game_turns=5,
        )
        state = {"game_over": False, "game_turn": 4}

        result = await env.game_over(state)
        assert result is False
        assert state["game_over"] is False, "game_turn < max should NOT set game_over"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_trajectory_step_records_agent_idx(self):
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        agent = _AgentContext(agent_idx=0)
        state = {
            "trajectory": [], "game_turn": 0,
            "_agents": [agent], "_current_agent_idx": 0,
            "manager": None,
        }

        step = {"completion": '{"move": "thunderbolt"}'}
        await env.add_trajectory_step(state, step)

        assert len(state["trajectory"]) == 1
        assert state["trajectory"][0]["extras"]["agent_idx"] == 0
        assert state["trajectory"][0]["extras"]["parsed_action"] == "None"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_trajectory_step_records_force_switch(self):
        """force_switch flag must be recorded in trajectory step extras."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        agent = _AgentContext(agent_idx=0)
        state = {
            "trajectory": [], "game_turn": 0,
            "_agents": [agent], "_current_agent_idx": 0,
            "manager": None,
        }

        step = {"completion": "whatever"}
        await env.add_trajectory_step(state, step)

        # No battle → force_switch should be False (not missing)
        assert "force_switch" in state["trajectory"][0]["extras"], (
            "force_switch must be present in trajectory step extras"
        )
        assert state["trajectory"][0]["extras"]["force_switch"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_win(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": True, "game_turn": 15,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 0}, "completion": "b"},
                {"extras": {"agent_idx": 0}, "completion": "c"},
            ],
        }

        await env.render_completion(state)

        assert state["reward"] == 1.0
        assert all(s["reward"] == 1.0 for s in state["trajectory"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_loss(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": False, "game_turn": 10,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 0}, "completion": "b"},
            ],
        }

        await env.render_completion(state)

        assert state["reward"] == 0.0
        assert all(s["reward"] == 0.0 for s in state["trajectory"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_win_and_loss_are_different(self):
        """Win reward and loss reward must be distinguishable."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        win_state = {
            "won": True, "game_turn": 10,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        loss_state = {
            "won": False, "game_turn": 10,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }

        await env.render_completion(win_state)
        await env.render_completion(loss_state)

        assert win_state["reward"] != loss_state["reward"], (
            "Win and loss rewards must be different"
        )
        assert win_state["trajectory"][0]["reward"] != loss_state["trajectory"][0]["reward"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_selfplay_opposite_rewards(self):
        """Self-play: P1 and P2 must get opposite rewards."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )

        state = {
            "won": True, "game_turn": 10,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},  # P1 step
                {"extras": {"agent_idx": 1}, "completion": "b"},  # P2 step
                {"extras": {"agent_idx": 0}, "completion": "c"},  # P1 step
                {"extras": {"agent_idx": 1}, "completion": "d"},  # P2 step
            ],
        }

        await env.render_completion(state)

        # P1 wins → P1 steps get 1.0, P2 steps get 0.0
        for step in state["trajectory"]:
            if step["extras"]["agent_idx"] == 0:
                assert step["reward"] == 1.0, (
                    f"P1 step should get 1.0 (win), got {step['reward']}"
                )
            else:
                assert step["reward"] == 0.0, (
                    f"P2 step should get 0.0 (loss), got {step['reward']}"
                )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_selfplay_p1_loses(self):
        """Self-play with P1 losing: rewards must be inverted."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
        )

        state = {
            "won": False, "game_turn": 10,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 1}, "completion": "b"},
            ],
        }

        await env.render_completion(state)

        # P1 loses → P1 gets 0.0, P2 gets 1.0
        assert state["trajectory"][0]["reward"] == 0.0, "P1 should get 0.0 (loss)"
        assert state["trajectory"][1]["reward"] == 1.0, "P2 should get 1.0 (win)"

    @pytest.mark.unit
    def test_passthrough_reward(self):
        from pokemon_rl.env import _passthrough_reward

        assert _passthrough_reward({"reward": 1.0}) == 1.0
        assert _passthrough_reward({"reward": 0.0}) == 0.0
        assert _passthrough_reward({}) == 0.0

    @pytest.mark.unit
    def test_invalid_play_mode_raises(self):
        """Unknown play_mode must raise ValueError."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError, match="play_mode"):
            PokemonBattleEnv(play_mode="unknown")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_won_none_metric(self):
        """won=None (draw/crash) → render_completion handles it gracefully."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": None, "truncated": False, "game_turn": 5,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        await env.render_completion(state)

        # reward_draw default is 0.0
        assert state["reward"] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_won_none_differs_from_loss(self):
        """won=None and won=False produce same default reward but are distinguishable by state['won']."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_draw=0.25,  # Make draw distinguishable from loss
        )

        none_state = {
            "won": None, "truncated": False, "game_turn": 5,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        loss_state = {
            "won": False, "truncated": False, "game_turn": 5,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }

        await env.render_completion(none_state)
        await env.render_completion(loss_state)

        assert none_state["reward"] != loss_state["reward"], (
            f"won=None ({none_state['reward']}) and won=False "
            f"({loss_state['reward']}) must be distinguishable with reward_draw=0.25"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_empty_trajectory_no_crash(self):
        """Empty trajectory must not crash render_completion."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": True, "game_turn": 0,
            "trajectory": [],
        }

        await env.render_completion(state)

        assert state["reward"] == 0.0  # no trajectory → default 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_standalone_without_adapter_raises(self):
        """run_standalone with no adapter must raise RuntimeError."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        with pytest.raises(RuntimeError, match="adapter"):
            await env.run_standalone()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_draw_uses_reward_draw(self):
        """N1: won=None uses configurable reward_draw (default 0.0)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": None, "truncated": True, "game_turn": 200,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 0}, "completion": "b"},
            ],
        }
        await env.render_completion(state)
        assert state["reward"] == 0.0, (
            f"Default reward_draw should be 0.0, got {state['reward']}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_custom_draw_reward(self):
        """N1: Custom reward_draw=0.5 gives 0.5 for won=None."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_draw=0.5,
        )
        state = {
            "won": None, "truncated": True, "game_turn": 200,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        await env.render_completion(state)
        assert state["reward"] == 0.5, (
            f"Custom reward_draw=0.5 should give 0.5, got {state['reward']}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_custom_terminal_rewards(self):
        """N1: Custom reward_win=10, reward_loss=-10, reward_draw=0.5."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_win=10.0, reward_loss=-10.0, reward_draw=0.5,
        )

        win_state = {
            "won": True, "truncated": False, "game_turn": 10,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        loss_state = {
            "won": False, "truncated": False, "game_turn": 10,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }
        draw_state = {
            "won": None, "truncated": True, "game_turn": 200,
            "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
        }

        await env.render_completion(win_state)
        await env.render_completion(loss_state)
        await env.render_completion(draw_state)

        assert win_state["reward"] == 10.0
        assert loss_state["reward"] == -10.0
        assert draw_state["reward"] == 0.5
        # All three must be distinguishable
        rewards = {win_state["reward"], loss_state["reward"], draw_state["reward"]}
        assert len(rewards) == 3, f"All 3 rewards must be distinct: {rewards}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_custom_rewards_no_inversion_bug(self):
        """N1: Self-play with reward_win=1, reward_loss=-1.

        CRITICAL: P2 should get reward_loss (-1.0), NOT 1.0 - 1.0 = 0.0.
        The old code used `1.0 - reward` which only works for [0,1].
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_win=1.0, reward_loss=-1.0,
        )

        state = {
            "won": True, "truncated": False, "game_turn": 10,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 1}, "completion": "b"},
            ],
        }
        await env.render_completion(state)

        assert state["trajectory"][0]["reward"] == 1.0, "P1 (winner) should get 1.0"
        assert state["trajectory"][1]["reward"] == -1.0, (
            f"P2 (loser) should get -1.0, got {state['trajectory'][1]['reward']}. "
            f"Old code used `1.0 - reward` which gives 0.0 — wrong for asymmetric rewards."
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_draw_both_get_draw_reward(self):
        """N1: Self-play draw gives both players reward_draw."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_draw=0.5,
        )
        state = {
            "won": None, "truncated": True, "game_turn": 200,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "a"},
                {"extras": {"agent_idx": 1}, "completion": "b"},
            ],
        }
        await env.render_completion(state)

        assert state["trajectory"][0]["reward"] == 0.5
        assert state["trajectory"][1]["reward"] == 0.5

    @pytest.mark.unit
    def test_compute_terminal_reward_defaults(self):
        """N1: _compute_terminal_reward with default config."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        assert env._compute_terminal_reward(True) == 1.0
        assert env._compute_terminal_reward(False) == 0.0
        assert env._compute_terminal_reward(None) == 0.0

    @pytest.mark.unit
    def test_compute_terminal_reward_custom(self):
        """N1: _compute_terminal_reward with custom config."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_win=10.0, reward_loss=-10.0, reward_draw=0.5,
        )
        assert env._compute_terminal_reward(True) == 10.0
        assert env._compute_terminal_reward(False) == -10.0
        assert env._compute_terminal_reward(None) == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_paths_use_compute_terminal_reward(self):
        """N1 CRITICAL: All reward paths must produce identical results.

        Tests that _assign_rewards gives same output regardless of
        self-play vs single for equivalent outcomes.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env_h = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_win=5.0, reward_loss=-5.0, reward_draw=0.5,
        )
        env_sp = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="self_play", observation_format="simple",
            reward_win=5.0, reward_loss=-5.0, reward_draw=0.5,
        )

        for won in [True, False, None]:
            h_state = {
                "won": won,
                "trajectory": [{"extras": {"agent_idx": 0}, "completion": "a"}],
            }
            sp_state = {
                "won": won,
                "trajectory": [
                    {"extras": {"agent_idx": 0}, "completion": "a"},
                    {"extras": {"agent_idx": 1}, "completion": "b"},
                ],
            }

            env_h._assign_rewards(h_state)
            env_sp._assign_rewards(sp_state)

            # P1 reward must match in both modes
            assert h_state["trajectory"][0]["reward"] == sp_state["trajectory"][0]["reward"], (
                f"won={won}: single P0 reward {h_state['trajectory'][0]['reward']} != "
                f"selfplay P0 reward {sp_state['trajectory'][0]['reward']}"
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_default_zero(self):
        """N1: Default step_reward_fn=None → no step_reward in extras."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        class FakeBattle:
            turn = 1
            force_switch = False

        class FakeTranslator:
            def parse_action(self, text, battle):
                return None
            def get_fallback_action(self, battle):
                class O:
                    message = "/choose default"
                return O()
            def extract_completion_text(self, completion):
                return str(completion)
            def extract_user_content(self, messages):
                return ""

        env.translator = FakeTranslator()
        agent = _AgentContext(agent_idx=0)
        agent.battle = FakeBattle()
        state = {
            "trajectory": [], "game_turn": 0,
            "_agents": [agent], "_current_agent_idx": 0,
            "manager": None,
        }
        await env.add_trajectory_step(state, {"completion": "x"})
        # No step_reward_fn → no step_reward key in extras
        assert "step_reward" not in state["trajectory"][0].get("extras", {})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_fn_called(self):
        """N1: Custom step_reward_fn is called with correct args."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        calls = []
        def spy(before, after, action, idx):
            calls.append((before, after, action, idx))
            return 0.42

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            step_reward_fn=spy,
        )

        class FakeBattle:
            turn = 1
            force_switch = False

        class FakeTranslator:
            def parse_action(self, text, battle):
                return None
            def get_fallback_action(self, battle):
                class O:
                    message = "/choose default"
                return O()
            def extract_completion_text(self, completion):
                return str(completion)
            def extract_user_content(self, messages):
                return ""

        env.translator = FakeTranslator()
        battle = FakeBattle()
        agent = _AgentContext(agent_idx=0)
        agent.battle = battle
        state = {
            "trajectory": [], "game_turn": 0,
            "_agents": [agent], "_current_agent_idx": 0,
            "manager": None,
        }
        await env.add_trajectory_step(state, {"completion": "x"})

        assert len(calls) == 1
        assert calls[0][0] is battle  # battle_before
        assert calls[0][3] == 0  # agent_idx
        assert state["trajectory"][0]["extras"]["step_reward"] == 0.42

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_separate_from_terminal(self):
        """N1: step_reward (in extras) and terminal reward are independent fields."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
            reward_win=1.0, step_reward_fn=lambda *_: 0.1,
        )
        state = {
            "won": True, "truncated": False, "game_turn": 5,
            "trajectory": [{"extras": {"agent_idx": 0, "step_reward": 0.1}, "completion": "a"}],
        }
        await env.render_completion(state)

        assert state["trajectory"][0]["reward"] == 1.0  # terminal
        assert state["trajectory"][0]["extras"]["step_reward"] == 0.1  # unchanged

    @pytest.mark.unit
    def test_passthrough_reward_allows_negative(self):
        """I2: _passthrough_reward must not zero out negative rewards."""
        from pokemon_rl.env import _passthrough_reward

        assert _passthrough_reward({"reward": -1.0}) == -1.0
        assert _passthrough_reward({"reward": 0}) == 0
        assert _passthrough_reward({"reward": 0.0}) == 0.0
        assert _passthrough_reward({}) == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_failure_tracked_in_agent(self):
        """C1 fix: Parse failures must be tracked in trajectory extras and agent context."""
        from pokemon_rl.env import PokemonBattleEnv, _AgentContext

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        # Need a mock translator that returns None from parse_action
        class FailTranslator:
            def parse_action(self, text, battle):
                return None

            def get_fallback_action(self, battle):
                class FakeOrder:
                    message = "/choose default"
                return FakeOrder()

            def battle_to_prompt(self, battle):
                return [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": "y"},
                ]

            def extract_completion_text(self, completion):
                return str(completion)

            def extract_user_content(self, messages):
                return ""

        env.translator = FailTranslator()

        class FakeBattle:
            turn = 1
            force_switch = False

        agent = _AgentContext(agent_idx=0)
        agent.battle = FakeBattle()
        state = {
            "trajectory": [], "game_turn": 0,
            "_agents": [agent], "_current_agent_idx": 0,
            "manager": None,
        }
        await env.add_trajectory_step(state, {"completion": "garbage"})
        assert state["trajectory"][0]["extras"]["parse_failed"] is True
        assert agent.parse_failure_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_setup_state_cleans_up_old_manager(self):
        """H7 fix: setup_state must close any existing manager."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )

        close_called = False

        class OldManager:
            async def close(self):
                nonlocal close_called
                close_called = True

        mock_new_manager = MagicMock()
        mock_battle = MagicMock()
        mock_new_manager.start_battle = AsyncMock(return_value=mock_battle)
        mock_new_manager.close = AsyncMock()

        state = {"manager": OldManager()}
        with patch("pokemon_rl.battle.BattleManager", return_value=mock_new_manager):
            await env.setup_state(state)
        assert close_called, "Old manager's close() should be called (H7 fix)"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_sets_completion_field(self):
        """render_completion must set state['completion'] from last trajectory step."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=8000,
            play_mode="single", observation_format="simple",
        )
        state = {
            "won": True, "game_turn": 5,
            "trajectory": [
                {"extras": {"agent_idx": 0}, "completion": "first"},
                {"extras": {"agent_idx": 0}, "completion": "last"},
            ],
        }
        await env.render_completion(state)
        assert state["completion"] == "last"

    @pytest.mark.unit
    def test_pokemon_rubric_passthrough(self):
        """PokemonRubric._passthrough_reward_sync returns state['reward'] or 0.0."""
        from pokemon_rl.env import PokemonRubric

        rubric = PokemonRubric()
        assert rubric._passthrough_reward_sync({"reward": 1.0}) == 1.0
        assert rubric._passthrough_reward_sync({"reward": -1.0}) == -1.0
        assert rubric._passthrough_reward_sync({"reward": 0.0}) == 0.0
        assert rubric._passthrough_reward_sync({}) == 0.0
        assert rubric._passthrough_reward_sync({"reward": None}) == 0.0


# ---- Integration tests: full game loop ----


@requires_poke_env
@requires_showdown
class TestEnvIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_standalone_simple(self, showdown_port):
        """Full game loop: env + adapter + simple translator."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )

        result = await env.run_standalone(adapter=adapter)

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
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )

        result = await env.run_standalone(adapter=adapter, action_fn=random_action)

        assert result["turns"] > 0
        assert "battle_tag" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trajectory_prompt_lengths(self, showdown_port):
        """Verify prompts are being generated and have reasonable length."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.env import PokemonBattleEnv

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )

        result = await env.run_standalone(adapter=adapter)

        for step in result["trajectory"]:
            assert step["prompt_length"] > 0
            assert step["prompt_messages"] == 2  # system + user

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_turn_by_turn_heuristic(self, showdown_port):
        """Turn-by-turn mode with heuristic opponent.

        Verifies:
        - Game starts and completes
        - Trajectory has steps with rewards
        - Decision count matches trajectory length
        - All steps are agent_idx=0 (only our player)
        - Rewards are consistent (all same value)
        """
        from pokemon_rl.env import PokemonBattleEnv
        from pokemon_rl.adapter import random_action

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
            opponent_type="random",
        )

        result = await env.run_turn_by_turn(action_fn=random_action)

        assert result["turns"] > 0, "Game should have turns"
        assert len(result["trajectory"]) > 0, "Trajectory should have steps"
        assert result["won"] in (True, False), (
            f"Expected True/False, got {result['won']}"
        )
        assert result["reward"] in (0.0, 1.0)
        assert result["decision_count"] == len(result["trajectory"]), (
            "decision_count should match trajectory length"
        )
        assert result["selfplay"] is False

        # Verify trajectory integrity
        for i, step in enumerate(result["trajectory"]):
            assert "reward" in step, f"Step {i} missing reward"
            assert step["reward"] == result["reward"], (
                f"Step {i} reward {step['reward']} != game reward {result['reward']}"
            )
            assert step["extras"]["agent_idx"] == 0, (
                f"Step {i} agent_idx should be 0, got {step['extras']['agent_idx']}"
            )
            assert "action" in step, f"Step {i} missing action"
            assert "force_switch" in step.get("extras", step), f"Step {i} missing force_switch"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_turn_by_turn_selfplay(self, showdown_port):
        """Self-play turn-by-turn mode.

        Verifies:
        - Both players generate trajectory steps
        - Rewards are opposite for the two players
        - Game completes with a winner
        - Force-switches may produce non-alternating agent indices

        NOTE: Agent indices do NOT strictly alternate 0,1,0,1 because
        force-switches only affect one player. E.g., after a faint:
        [0,1, 0, 0,1, ...] where the extra 0 is a force-switch.
        """
        from pokemon_rl.env import PokemonBattleEnv
        from pokemon_rl.adapter import random_action

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
        )

        result = await env.run_turn_by_turn(action_fn=random_action)

        assert result["turns"] > 0, "Game should have turns"
        assert len(result["trajectory"]) > 0, "Trajectory should have steps"
        assert result["won"] in (True, False), (
            f"Expected True/False, got {result['won']}"
        )
        assert result["selfplay"] is True

        # Verify both players have steps
        p1_steps = [s for s in result["trajectory"] if s["extras"]["agent_idx"] == 0]
        p2_steps = [s for s in result["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p1_steps) > 0, "P1 should have trajectory steps"
        assert len(p2_steps) > 0, "P2 should have trajectory steps"

        # Verify agent indices are only 0 or 1 (no other values)
        for i, step in enumerate(result["trajectory"]):
            assert step["extras"]["agent_idx"] in (0, 1), (
                f"Step {i}: agent_idx must be 0 or 1, got {step['extras']['agent_idx']}"
            )

        # Verify opposite rewards
        if result["won"]:
            expected_p1_reward = 1.0
            expected_p2_reward = 0.0
        else:
            expected_p1_reward = 0.0
            expected_p2_reward = 1.0

        for step in p1_steps:
            assert step["reward"] == expected_p1_reward, (
                f"P1 reward should be {expected_p1_reward}, got {step['reward']}"
            )
        for step in p2_steps:
            assert step["reward"] == expected_p2_reward, (
                f"P2 reward should be {expected_p2_reward}, got {step['reward']}"
            )

        # P1 and P2 rewards must be different (one wins, one loses)
        assert p1_steps[0]["reward"] != p2_steps[0]["reward"], (
            "Self-play rewards must be opposite"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_turn_by_turn_trajectory_has_actions(self, showdown_port):
        """Every trajectory step must have a non-empty action string."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )

        result = await env.run_turn_by_turn()

        for i, step in enumerate(result["trajectory"]):
            assert "action" in step, f"Step {i} missing action"
            assert len(step["action"]) > 0, (
                f"Step {i} action is empty string"
            )
            # Action should be a poke-env message like "/choose move thunderbolt"
            # or "/choose switch pikachu"
            assert step["action"].startswith("/choose") or step["action"].startswith("/"), (
                f"Step {i} action doesn't look like a poke-env command: {step['action']}"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_turn_by_turn_trajectory_turns_monotonic(self, showdown_port):
        """Turn numbers must be monotonically non-decreasing.

        Force-switches can repeat the same turn number (they don't advance
        the game turn), but turns must never go backwards.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )

        result = await env.run_turn_by_turn()

        turns = [s["turn"] for s in result["trajectory"]]
        for i in range(1, len(turns)):
            assert turns[i] >= turns[i - 1], (
                f"Turn sequence not monotonic at index {i}: ...{turns[max(0,i-2):i+2]}..."
            )
        assert all(t >= 1 for t in turns), f"All turns should be >= 1, got: {turns}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hooks_heuristic_integration(self, showdown_port):
        """Full hooks cycle with real Showdown (not the run_turn_by_turn shortcut).

        Tests the ACTUAL path that verifiers will use:
        setup_state → game_over check → (get_prompt → add_step) x N → render_completion.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
            opponent_type="random",
        )

        state = await env.setup_state({})

        assert state["manager"] is not None
        assert state["_agents"][0].battle is not None
        assert state["game_over"] is False

        step_count = 0
        while not state["game_over"] and step_count < 300:
            is_over = await env.game_over(state)
            if is_over:
                break
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            assert len(prompt) == 2
            assert prompt[0]["role"] == "system"

            # Garbage completion → triggers fallback action
            step = {"completion": '{"move": "nonexistent"}'}
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        assert step_count > 0, "Game should have at least 1 step"
        assert state["reward"] in (0.0, 1.0)
        assert len(state["trajectory"]) == step_count

        for i, s in enumerate(state["trajectory"]):
            assert "extras" in s, f"Step {i}: missing extras"
            assert "agent_idx" in s["extras"], f"Step {i}: missing agent_idx in extras"
            assert "parsed_action" in s["extras"], f"Step {i}: missing parsed_action in extras"
            assert "force_switch" in s["extras"], f"Step {i}: missing force_switch in extras"
            assert "reward" in s, f"Step {i}: missing reward"
            assert s["extras"]["agent_idx"] == 0, f"Step {i}: single mode = agent 0"
