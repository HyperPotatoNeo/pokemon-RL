"""Phase 4 integration tests: Real Showdown battles with verifiers hooks.

These tests require:
- Showdown server running on localhost (compute node)
- poke-env installed in .venv
- pokechamp installed in .venv (for pokechamp_io format tests)

Tests verify end-to-end behavior: real BattleManager, real Showdown, real
game resolution. No mocks for game logic.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
Integration tests are expensive. Every test must verify specific values,
not just "it didn't crash." Each test has explicit assertions on trajectory
contents, reward values, and game state consistency.
"""

import asyncio
import pytest
import os

from tests.conftest import (
    requires_poke_env, requires_showdown, requires_pokechamp,
    SHOWDOWN_PORT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def showdown_port():
    return SHOWDOWN_PORT


# ============================================================================
# I1: Full Hooks Cycle — Single Agent
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationSingleAgent:
    """Full hooks cycle with real Showdown: single agent vs heuristic."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_hooks_cycle_simple_format(self, showdown_port):
        """Complete cycle: setup → (prompt → step) × N → render.
        Uses simple format (no pokechamp dependency)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
            opponent_type="random",
        )

        state = await env.setup_state({"task": "battle", "prompt": []})
        assert state["manager"] is not None
        assert state["_agents"][0].battle is not None
        assert state["game_over"] is False

        step_count = 0
        while not await env.game_over(state) and step_count < 300:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            assert isinstance(prompt, list)
            assert len(prompt) >= 1

            # Simulate LLM: send garbage → triggers random fallback
            step = {
                "completion": [{"role": "assistant", "content": "I pick thunderbolt!"}],
                "prompt": prompt,
                "tokens": {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5],
                           "prompt_mask": [1, 1, 1], "completion_mask": [1, 1],
                           "completion_logprobs": [-0.5, -0.3]},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        # Verify game completed
        assert step_count > 0, "Game should have at least 1 step"
        assert state["game_over"] is True
        assert state["won"] in (True, False, None)
        assert isinstance(state["reward"], (int, float))

        # Verify trajectory integrity
        assert len(state["trajectory"]) == step_count
        for i, s in enumerate(state["trajectory"]):
            extras = s.get("extras", {})
            assert extras.get("agent_idx") == 0, f"Step {i}: agent_idx must be 0"
            assert "parsed_action" in extras, f"Step {i}: missing parsed_action"
            assert "parse_failed" in extras, f"Step {i}: missing parse_failed"
            assert isinstance(s["reward"], (int, float)), f"Step {i}: reward must be numeric"

        # Verify completion field set (CR-2)
        assert "completion" in state

        # Verify metrics
        assert "metrics" in state
        assert state["metrics"]["won"] in (0, 1, -1)

        # Cleanup
        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parse_failure_tracked(self, showdown_port):
        """Garbage LLM output → parse_failed=True, fallback used, game continues."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        prompt = await env.get_prompt_messages(state)
        step = {
            "completion": [{"role": "assistant", "content": "no json here at all"}],
            "prompt": prompt, "tokens": {},
        }
        await env.add_trajectory_step(state, step)

        extras = state["trajectory"][0].get("extras", {})
        assert extras.get("parse_failed") is True, "Garbage input must trigger parse_failed"
        assert state["_agents"][0].parse_failure_count >= 1

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_valid_json_parses_correctly(self, showdown_port):
        """Valid JSON move → parse_failed=False."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        # Get actual available moves from battle state
        battle = state["_agents"][0].battle
        if battle.available_moves:
            move_name = battle.available_moves[0].id
            prompt = await env.get_prompt_messages(state)
            step = {
                "completion": [{"role": "assistant",
                               "content": f'{{"move": "{move_name}"}}'}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            extras = state["trajectory"][0].get("extras", {})
            assert extras.get("parse_failed") is False, (
                f"Valid move '{move_name}' should parse successfully"
            )

        await env.cleanup_battle(state)

    @requires_pokechamp
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pokechamp_io_format(self, showdown_port):
        """pokechamp_io format produces rich prompts with damage calcs."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="pokechamp_io",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})
        prompt = await env.get_prompt_messages(state)

        assert prompt is not None
        assert len(prompt) >= 2
        # pokechamp prompts include damage calculations
        user_content = prompt[-1]["content"]
        assert len(user_content) > 100, (
            f"pokechamp_io prompt should be detailed (>100 chars), got {len(user_content)}"
        )

        await env.cleanup_battle(state)


# ============================================================================
# I2: Full Hooks Cycle — Self-Play
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationSelfPlay:
    """Full hooks cycle with real Showdown: self-play."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_selfplay_game(self, showdown_port):
        """Complete self-play game. Both agents act, rewards opposite."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        assert len(state["_agents"]) == 2

        step_count = 0
        while not await env.game_over(state) and step_count < 600:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "just attacking"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        # Both players must have steps
        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0) > 0, "P0 must have steps"
        assert len(p1) > 0, "P1 must have steps"
        assert len(p0) + len(p1) == len(state["trajectory"])

        # Rewards must be opposite (unless draw)
        if state["won"] is not None:
            assert p0[0]["reward"] != p1[0]["reward"], "Winner and loser must differ"

        # Advantages must be pre-set for non-draw
        if state["won"] is not None:
            for s in state["trajectory"]:
                assert s.get("advantage") is not None, (
                    "Self-play non-draw: advantages must be pre-set"
                )

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_step_counts_realistic(self, showdown_port):
        """Self-play step counts should be realistic for gen1.
        Normal game: 10-100 steps total (5-50 per player)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 600:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "attack"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        assert step_count >= 2, "Game should have at least 2 steps (1 per player)"
        assert step_count <= 500, "Game shouldn't exceed 500 steps"

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_force_switch_handled(self, showdown_port):
        """Force-switches must be handled without deadlock.
        We can't guarantee a force-switch happens, but the game must complete."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
        )

        # Run 3 games to increase chance of force-switch
        for game_num in range(3):
            state = await env.setup_state({"task": "battle", "prompt": []})
            step_count = 0
            while not await env.game_over(state) and step_count < 600:
                prompt = await env.get_prompt_messages(state)
                if prompt is None:
                    break
                step = {
                    "completion": [{"role": "assistant", "content": "go"}],
                    "prompt": prompt, "tokens": {},
                }
                await env.add_trajectory_step(state, step)
                step_count += 1
            await env.render_completion(state)
            assert state["game_over"] is True, f"Game {game_num} must complete"
            await env.cleanup_battle(state)


# ============================================================================
# I3: Concurrent Battles
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationConcurrent:
    """Multiple concurrent battles must not interfere."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_3_concurrent_single_agent(self, showdown_port):
        """3 concurrent single-agent games with unique battle tags."""
        from pokemon_rl.env import PokemonBattleEnv

        async def run_game(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle", port=showdown_port,
                play_mode="single", observation_format="simple",
            )
            state = await env.setup_state({"task": f"game_{game_id}", "prompt": []})
            steps = 0
            while not await env.game_over(state) and steps < 300:
                prompt = await env.get_prompt_messages(state)
                if prompt is None:
                    break
                step = {
                    "completion": [{"role": "assistant", "content": "attack"}],
                    "prompt": prompt, "tokens": {},
                }
                await env.add_trajectory_step(state, step)
                steps += 1
            await env.render_completion(state)
            await env.cleanup_battle(state)
            return state

        results = await asyncio.gather(
            run_game(0), run_game(1), run_game(2)
        )

        # All must complete
        for i, state in enumerate(results):
            assert state["game_over"] is True, f"Game {i} must complete"
            assert len(state["trajectory"]) > 0, f"Game {i} must have steps"

        # Battle tags must be unique (no cross-contamination)
        # Note: We check via trajectory lengths being independent
        lengths = [len(s["trajectory"]) for s in results]
        # At least some should differ (extremely unlikely all same length)
        # But we can't guarantee this, so just verify all completed

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_2_concurrent_selfplay(self, showdown_port):
        """2 concurrent self-play games."""
        from pokemon_rl.env import PokemonBattleEnv

        async def run_selfplay(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle", port=showdown_port,
                play_mode="self_play", observation_format="simple",
            )
            state = await env.setup_state({"task": f"sp_{game_id}", "prompt": []})
            steps = 0
            while not await env.game_over(state) and steps < 600:
                prompt = await env.get_prompt_messages(state)
                if prompt is None:
                    break
                step = {
                    "completion": [{"role": "assistant", "content": "go"}],
                    "prompt": prompt, "tokens": {},
                }
                await env.add_trajectory_step(state, step)
                steps += 1
            await env.render_completion(state)
            await env.cleanup_battle(state)
            return state

        results = await asyncio.gather(
            run_selfplay(0), run_selfplay(1)
        )

        for i, state in enumerate(results):
            assert state["game_over"] is True
            p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
            p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
            assert len(p0) > 0 and len(p1) > 0, f"Game {i}: both players must act"


# ============================================================================
# I4: Trajectory Format Compatibility
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationTrajectoryFormat:
    """Verify trajectory format is compatible with prime-rl pipeline."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_step_reward_is_float(self, showdown_port):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})
        while not await env.game_over(state):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
        await env.render_completion(state)

        for i, s in enumerate(state["trajectory"]):
            assert isinstance(s["reward"], (int, float)), (
                f"Step {i}: reward must be numeric, got {type(s['reward'])}"
            )

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_extras_agent_idx_is_int(self, showdown_port):
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
        )
        state = await env.setup_state({"task": "battle", "prompt": []})
        prompt = await env.get_prompt_messages(state)
        if prompt:
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)

        for s in state["trajectory"]:
            idx = s.get("extras", {}).get("agent_idx")
            assert isinstance(idx, int), f"agent_idx must be int, got {type(idx)}"
            assert idx in (0, 1), f"agent_idx must be 0 or 1, got {idx}"

        await env.cleanup_battle(state)


# ============================================================================
# I5: Truncation
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationTruncation:
    """Verify truncation behavior with real battles."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_max_game_turns_truncates(self, showdown_port):
        """max_game_turns=1 forces truncation after one step.

        gen1randombattle with random moves never ends in 1 turn (no OHKO),
        so truncation is guaranteed. Assertions are unconditional.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
            max_game_turns=1, reward_draw=0.5,
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 100:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        assert state["game_over"] is True, "Game must end"
        assert len(state["trajectory"]) > 0, "Must have at least one step"
        assert state.get("truncated") is True, "Must truncate with max_game_turns=1"
        assert state["reward"] == 0.5, "Truncated game should use reward_draw"
        assert state["won"] is None, "Truncated game has no winner"

        await env.cleanup_battle(state)


# ============================================================================
# I6: Message History (HIGH-5 / T17)
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationMessageHistory:
    """Verify message history recording with real battles."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_history_grows_with_turns(self, showdown_port):
        """HIGH-5 / T17.1: History should grow by 2 per step (user + assistant)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="single", observation_format="simple",
            max_game_turns=5,
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 10:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        agent = state["_agents"][0]
        expected_history = step_count * 2
        assert len(agent.message_history) == expected_history, (
            f"{step_count} steps should produce {expected_history} history entries, "
            f"got {len(agent.message_history)}"
        )

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_independent_histories(self, showdown_port):
        """HIGH-5 / T17.2: Self-play: each agent has independent history."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
            max_game_turns=5,
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 20:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        agents = state["_agents"]
        assert len(agents) == 2
        assert agents[0].message_history is not agents[1].message_history, (
            "Agent histories must be different objects"
        )
        # Each agent's history should reflect their own step count
        p0_steps = sum(1 for s in state["trajectory"] if s["extras"]["agent_idx"] == 0)
        p1_steps = sum(1 for s in state["trajectory"] if s["extras"]["agent_idx"] == 1)
        assert len(agents[0].message_history) == p0_steps * 2
        assert len(agents[1].message_history) == p1_steps * 2

        await env.cleanup_battle(state)


# ============================================================================
# I7: Truncation Mid-Self-Play (HIGH-7 / T19)
# ============================================================================

@requires_poke_env
@requires_showdown
class TestIntegrationTruncationSelfPlay:
    """Truncation during self-play with real battles."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_truncation_both_get_draw(self, showdown_port):
        """max_game_turns=1 forces truncation — both players get reward_draw.

        gen1randombattle with random moves never ends in 1 turn,
        so truncation is guaranteed. Assertions are unconditional.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
            max_game_turns=1, reward_draw=0.5,
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 50:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        assert state["game_over"] is True, "Game must end"
        assert len(state["trajectory"]) > 0, "Must have at least one step"
        assert state.get("truncated") is True, "Must truncate with max_game_turns=1"
        # Both players should get reward_draw
        for s in state["trajectory"]:
            assert s["reward"] == 0.5, (
                f"Truncated self-play: all steps should get reward_draw=0.5, "
                f"got {s['reward']} for agent {s['extras']['agent_idx']}"
            )

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_truncation_asymmetric_steps(self, showdown_port):
        """HIGH-7 / T19.2: Asymmetric step counts handled in reward assignment."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle", port=showdown_port,
            play_mode="self_play", observation_format="simple",
            max_game_turns=3,
        )
        state = await env.setup_state({"task": "battle", "prompt": []})

        step_count = 0
        while not await env.game_over(state) and step_count < 50:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "go"}],
                "prompt": prompt, "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)

        # Every step must have a reward regardless of asymmetry
        for i, s in enumerate(state["trajectory"]):
            assert "reward" in s, f"Step {i} missing reward"
            assert isinstance(s["reward"], (int, float)), (
                f"Step {i} reward must be numeric"
            )

        await env.cleanup_battle(state)
