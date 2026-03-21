"""Tests for Layer 4: PokemonBattleEnv.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    1. REWARD CORRECTNESS: Test that wins get 1.0, losses get 0.0, AND that
       they are NOT the same value (distinguishable outcomes).
    2. TRAJECTORY INTEGRITY: Verify step counts match, player indices are
       correct, force_switch flags are present, turn numbers are recorded.
    3. SELF-PLAY SYMMETRY: Both players get different states. Rewards are
       opposite. Player indices alternate correctly.
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
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = await env.setup_state({})

        assert state["trajectory"] == []
        assert state["game_over"] is False
        assert state["turn"] == 0
        assert state["decision_count"] == 0
        assert state["won"] is None
        assert state["truncated"] is False
        assert state["parse_failure_count"] == 0
        assert state["battle"] is None
        assert state["manager"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_game_over_returns_none(self):
        """Game over → get_prompt must return None (not empty list)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {"game_over": True, "turn": 10}

        result = await env.get_prompt_messages(state)
        assert result is None, "game_over=True must return None, not empty"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_not_game_over_but_no_battle(self):
        """Not game over but no battle → return None (not crash)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {"game_over": False, "turn": 0, "battle": None}

        result = await env.get_prompt_messages(state)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_max_turns_triggers_game_over(self):
        """Reaching max_game_turns must set game_over, truncated, and return None."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None, max_game_turns=5)
        state = {"game_over": False, "turn": 5, "battle": None, "won": None}

        result = await env.get_prompt_messages(state)
        assert result is None
        assert state["game_over"] is True, "Should set game_over flag"
        assert state["truncated"] is True, "Should set truncated flag (C2 fix)"
        assert state["won"] is None, "Truncation should not set won=False"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_prompt_below_max_turns_does_not_end(self):
        """Below max_game_turns should NOT set game_over."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None, max_game_turns=5)
        state = {"game_over": False, "turn": 4, "battle": None}

        # No battle set, so returns None, but game_over should NOT be set
        await env.get_prompt_messages(state)
        assert state["game_over"] is False, "turn < max should NOT set game_over"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_trajectory_step_increments_decision_count(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "trajectory": [], "turn": 0, "decision_count": 0,
            "battle": None, "manager": None
        }

        step = {"completion": '{"move": "thunderbolt"}'}
        await env.add_trajectory_step(state, step)

        assert len(state["trajectory"]) == 1
        assert state["decision_count"] == 1
        assert state["trajectory"][0]["player_idx"] == 0
        assert state["trajectory"][0]["parsed_action"] == "no_battle"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_trajectory_step_records_force_switch(self):
        """force_switch flag must be recorded in trajectory step."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "trajectory": [], "turn": 0, "decision_count": 0,
            "battle": None, "manager": None
        }

        step = {"completion": "whatever"}
        await env.add_trajectory_step(state, step)

        # No battle → force_switch should be False (not missing)
        assert "force_switch" in state["trajectory"][0], (
            "force_switch must be present in trajectory step"
        )
        assert state["trajectory"][0]["force_switch"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_win(self):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": True, "turn": 15, "decision_count": 15,
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
            "won": False, "turn": 10, "decision_count": 10,
            "trajectory": [{"player_idx": 0}, {"player_idx": 0}],
        }

        await env.render_completion(state)

        assert state["reward"] == 0.0
        assert all(s["reward"] == 0.0 for s in state["trajectory"])
        assert state["metrics"]["won"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_win_and_loss_are_different(self):
        """Win reward and loss reward must be distinguishable."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)

        win_state = {
            "won": True, "turn": 10, "decision_count": 10,
            "trajectory": [{"player_idx": 0}],
        }
        loss_state = {
            "won": False, "turn": 10, "decision_count": 10,
            "trajectory": [{"player_idx": 0}],
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
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        state = {
            "won": True, "turn": 10, "decision_count": 20,
            "trajectory": [
                {"player_idx": 0},  # P1 step
                {"player_idx": 1},  # P2 step
                {"player_idx": 0},  # P1 step
                {"player_idx": 1},  # P2 step
            ],
        }

        await env.render_completion(state)

        # P1 wins → P1 steps get 1.0, P2 steps get 0.0
        for step in state["trajectory"]:
            if step["player_idx"] == 0:
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
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )

        state = {
            "won": False, "turn": 10, "decision_count": 20,
            "trajectory": [
                {"player_idx": 0},
                {"player_idx": 1},
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
    def test_invalid_control_mode_raises(self):
        """Unknown control_mode must raise ValueError."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError, match="control_mode"):
            PokemonBattleEnv(control_mode="unknown")

    @pytest.mark.unit
    def test_invalid_opponent_mode_raises(self):
        """Unknown opponent_mode must raise ValueError."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError, match="opponent_mode"):
            PokemonBattleEnv(opponent_mode="unknown")

    @pytest.mark.unit
    def test_selfplay_requires_turn_by_turn(self):
        """Self-play with full_battle mode must raise ValueError."""
        from pokemon_rl.env import PokemonBattleEnv

        with pytest.raises(ValueError, match="Self-play"):
            PokemonBattleEnv(
                control_mode="full_battle",
                opponent_mode="self_play",
            )

    @pytest.mark.unit
    def test_selfplay_with_turn_by_turn_ok(self):
        """Self-play with turn_by_turn mode should NOT raise."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
        )
        assert env.opponent_mode == "self_play"
        assert env.control_mode == "turn_by_turn"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_won_none_metric_negative_one(self):
        """won=None (draw/crash) → metrics['won'] must be -1, not 0."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": None, "truncated": False, "turn": 5, "decision_count": 5,
            "trajectory": [{"player_idx": 0}],
        }
        await env.render_completion(state)

        assert state["metrics"]["won"] == -1, (
            f"won=None should give metrics['won']=-1, got {state['metrics']['won']}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_won_none_differs_from_loss(self):
        """won=None and won=False must be distinguishable in metrics."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)

        none_state = {
            "won": None, "truncated": False, "turn": 5, "decision_count": 5,
            "trajectory": [{"player_idx": 0}],
        }
        loss_state = {
            "won": False, "truncated": False, "turn": 5, "decision_count": 5,
            "trajectory": [{"player_idx": 0}],
        }

        await env.render_completion(none_state)
        await env.render_completion(loss_state)

        assert none_state["metrics"]["won"] != loss_state["metrics"]["won"], (
            f"won=None ({none_state['metrics']['won']}) and won=False "
            f"({loss_state['metrics']['won']}) must be distinguishable"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_empty_trajectory_no_crash(self):
        """Empty trajectory must not crash render_completion."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": True, "turn": 0, "decision_count": 0,
            "trajectory": [],
        }

        await env.render_completion(state)

        assert state["reward"] == 1.0
        assert state["metrics"]["trajectory_length"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_standalone_without_adapter_raises(self):
        """run_standalone with no adapter must raise RuntimeError."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        with pytest.raises(RuntimeError, match="adapter"):
            await env.run_standalone()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_draw_uses_reward_draw(self):
        """N1: won=None uses configurable reward_draw (default 0.0)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        state = {
            "won": None, "truncated": True, "turn": 200, "decision_count": 200,
            "trajectory": [{"player_idx": 0}, {"player_idx": 0}],
        }
        await env.render_completion(state)
        assert state["reward"] == 0.0, (
            f"Default reward_draw should be 0.0, got {state['reward']}"
        )
        assert state["metrics"]["truncated"] == 1
        assert state["metrics"]["won"] == -1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_render_completion_custom_draw_reward(self):
        """N1: Custom reward_draw=0.5 gives 0.5 for won=None."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None, reward_draw=0.5)
        state = {
            "won": None, "truncated": True, "turn": 200, "decision_count": 200,
            "trajectory": [{"player_idx": 0}],
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
            adapter=None, translator=None,
            reward_win=10.0, reward_loss=-10.0, reward_draw=0.5,
        )

        win_state = {
            "won": True, "truncated": False, "turn": 10, "decision_count": 10,
            "trajectory": [{"player_idx": 0}],
        }
        loss_state = {
            "won": False, "truncated": False, "turn": 10, "decision_count": 10,
            "trajectory": [{"player_idx": 0}],
        }
        draw_state = {
            "won": None, "truncated": True, "turn": 200, "decision_count": 200,
            "trajectory": [{"player_idx": 0}],
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
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
            reward_win=1.0,
            reward_loss=-1.0,
        )

        state = {
            "won": True, "truncated": False, "turn": 10, "decision_count": 20,
            "trajectory": [
                {"player_idx": 0},
                {"player_idx": 1},
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
            translator=None,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
            reward_draw=0.5,
        )
        state = {
            "won": None, "truncated": True, "turn": 200, "decision_count": 400,
            "trajectory": [
                {"player_idx": 0},
                {"player_idx": 1},
            ],
        }
        await env.render_completion(state)

        assert state["trajectory"][0]["reward"] == 0.5
        assert state["trajectory"][1]["reward"] == 0.5

    @pytest.mark.unit
    def test_compute_terminal_reward_defaults(self):
        """N1: _compute_terminal_reward with default config."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)
        assert env._compute_terminal_reward(True) == 1.0
        assert env._compute_terminal_reward(False) == 0.0
        assert env._compute_terminal_reward(None) == 0.0

    @pytest.mark.unit
    def test_compute_terminal_reward_custom(self):
        """N1: _compute_terminal_reward with custom config."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            adapter=None, translator=None,
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
        self-play vs heuristic for equivalent outcomes.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env_h = PokemonBattleEnv(
            adapter=None, translator=None,
            reward_win=5.0, reward_loss=-5.0, reward_draw=0.5,
        )
        env_sp = PokemonBattleEnv(
            translator=None, control_mode="turn_by_turn",
            opponent_mode="self_play",
            reward_win=5.0, reward_loss=-5.0, reward_draw=0.5,
        )

        for won in [True, False, None]:
            h_traj = [{"player_idx": 0}]
            h_reward = env_h._assign_rewards(h_traj, won)
            sp_traj = [{"player_idx": 0}, {"player_idx": 1}]
            sp_reward = env_sp._assign_rewards(sp_traj, won)

            # P1 reward must match in both modes
            assert h_traj[0]["reward"] == sp_traj[0]["reward"], (
                f"won={won}: heuristic P0 reward {h_traj[0]['reward']} != "
                f"selfplay P0 reward {sp_traj[0]['reward']}"
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_default_zero(self):
        """N1: Default step_reward_fn=None → step_reward=0.0."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)

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

        env.translator = FakeTranslator()
        state = {
            "trajectory": [], "turn": 0, "decision_count": 0,
            "battle": FakeBattle(), "manager": None,
            "parse_failure_count": 0,
        }
        await env.add_trajectory_step(state, {"completion": "x"})
        assert state["trajectory"][0]["step_reward"] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_fn_called(self):
        """N1: Custom step_reward_fn is called with correct args."""
        from pokemon_rl.env import PokemonBattleEnv

        calls = []
        def spy(before, after, action, idx):
            calls.append((before, after, action, idx))
            return 0.42

        env = PokemonBattleEnv(
            adapter=None, translator=None, step_reward_fn=spy,
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

        env.translator = FakeTranslator()
        battle = FakeBattle()
        state = {
            "trajectory": [], "turn": 0, "decision_count": 0,
            "battle": battle, "manager": None,
            "parse_failure_count": 0,
        }
        await env.add_trajectory_step(state, {"completion": "x"})

        assert len(calls) == 1
        assert calls[0][0] is battle  # battle_before
        assert calls[0][3] == 0  # player_idx
        assert state["trajectory"][0]["step_reward"] == 0.42

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_reward_separate_from_terminal(self):
        """N1: step_reward and terminal reward are independent fields."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            adapter=None, translator=None,
            reward_win=1.0, step_reward_fn=lambda *_: 0.1,
        )
        state = {
            "won": True, "truncated": False, "turn": 5, "decision_count": 5,
            "trajectory": [{"player_idx": 0, "step_reward": 0.1}],
        }
        await env.render_completion(state)

        assert state["trajectory"][0]["reward"] == 1.0  # terminal
        assert state["trajectory"][0]["step_reward"] == 0.1  # unchanged

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
    async def test_parse_failure_tracked_in_state(self):
        """C1 fix: Parse failures must be tracked in trajectory and state."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)

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

        env.translator = FailTranslator()

        class FakeBattle:
            turn = 1
            force_switch = False

        state = {
            "trajectory": [], "turn": 0, "decision_count": 0,
            "battle": FakeBattle(), "manager": None,
            "parse_failure_count": 0,
        }
        await env.add_trajectory_step(state, {"completion": "garbage"})
        assert state["trajectory"][0]["parse_failed"] is True
        assert state["parse_failure_count"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_setup_state_cleans_up_old_manager(self):
        """H7 fix: setup_state must close any existing manager."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(adapter=None, translator=None)

        close_called = False

        class OldManager:
            async def close(self):
                nonlocal close_called
                close_called = True

        state = {"manager": OldManager()}
        await env.setup_state(state)
        assert close_called, "Old manager's close() should be called (H7 fix)"


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

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_turn_by_turn_heuristic(self, showdown_port):
        """Turn-by-turn mode with heuristic opponent.

        Verifies:
        - Game starts and completes
        - Trajectory has steps with rewards
        - Decision count matches trajectory length
        - All steps are player_idx=0 (only our player)
        - Rewards are consistent (all same value)
        """
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv
        from pokemon_rl.adapter import random_action

        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(
            translator=translator,
            control_mode="turn_by_turn",
            port=showdown_port,
            battle_format="gen1randombattle",
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
            assert step["player_idx"] == 0, (
                f"Step {i} player_idx should be 0, got {step['player_idx']}"
            )
            assert "action" in step, f"Step {i} missing action"
            assert "force_switch" in step, f"Step {i} missing force_switch"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_turn_by_turn_selfplay(self, showdown_port):
        """Self-play turn-by-turn mode.

        Verifies:
        - Both players generate trajectory steps
        - Rewards are opposite for the two players
        - Game completes with a winner
        - Force-switches may produce non-alternating player indices

        NOTE: Player indices do NOT strictly alternate 0,1,0,1 because
        force-switches only affect one player. E.g., after a faint:
        [0,1, 0, 0,1, ...] where the extra 0 is a force-switch.
        """
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv
        from pokemon_rl.adapter import random_action

        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(
            translator=translator,
            control_mode="turn_by_turn",
            opponent_mode="self_play",
            port=showdown_port,
            battle_format="gen1randombattle",
        )

        result = await env.run_turn_by_turn(action_fn=random_action)

        assert result["turns"] > 0, "Game should have turns"
        assert len(result["trajectory"]) > 0, "Trajectory should have steps"
        assert result["won"] in (True, False), (
            f"Expected True/False, got {result['won']}"
        )
        assert result["selfplay"] is True

        # Verify both players have steps
        p1_steps = [s for s in result["trajectory"] if s["player_idx"] == 0]
        p2_steps = [s for s in result["trajectory"] if s["player_idx"] == 1]
        assert len(p1_steps) > 0, "P1 should have trajectory steps"
        assert len(p2_steps) > 0, "P2 should have trajectory steps"

        # Verify player indices are only 0 or 1 (no other values)
        for i, step in enumerate(result["trajectory"]):
            assert step["player_idx"] in (0, 1), (
                f"Step {i}: player_idx must be 0 or 1, got {step['player_idx']}"
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
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(
            translator=translator,
            control_mode="turn_by_turn",
            port=showdown_port,
            battle_format="gen1randombattle",
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
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(
            translator=translator,
            control_mode="turn_by_turn",
            port=showdown_port,
            battle_format="gen1randombattle",
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
        setup_state → (get_prompt → add_step) × N → render_completion.
        """
        from pokemon_rl.translator import StateTranslator
        from pokemon_rl.env import PokemonBattleEnv

        translator = StateTranslator(format_style="simple")
        env = PokemonBattleEnv(
            translator=translator,
            control_mode="turn_by_turn",
            port=showdown_port,
            battle_format="gen1randombattle",
            opponent_type="random",
        )

        state = await env.setup_state({})

        assert state["manager"] is not None
        assert state["battle"] is not None
        assert state["game_over"] is False

        step_count = 0
        while not state["game_over"] and step_count < 300:
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
            assert "player_idx" in s, f"Step {i}: missing player_idx"
            assert "parsed_action" in s, f"Step {i}: missing parsed_action"
            assert "force_switch" in s, f"Step {i}: missing force_switch"
            assert "reward" in s, f"Step {i}: missing reward"
            assert s["player_idx"] == 0, f"Step {i}: heuristic = player 0"
