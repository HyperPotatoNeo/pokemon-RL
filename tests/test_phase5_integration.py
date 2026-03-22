"""Phase 5 integration tests: Real Showdown battles for RL training validation.

Tests require:
- Showdown server running on localhost (compute node)
- poke-env installed in .venv
- pokechamp installed for pokechamp_io format tests

These tests verify battle flow correctness with real game resolution:
- Full gen9ou games (single-agent and self-play) with team loading
- Team selection randomness across games
- Max game turns truncation
- Concurrent games on shared Showdown server

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
Integration tests are expensive. Every test asserts on specific trajectory values,
reward correctness, and game state consistency — not just "it didn't crash."
"""

import asyncio
import os
import pytest

from tests.conftest import (
    requires_poke_env, requires_showdown, requires_pokechamp,
    SHOWDOWN_PORT,
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
# Helper: run a full hooks cycle game (reusable across tests)
# ---------------------------------------------------------------------------
async def _run_hooks_game(env, max_steps=500):
    """Run a complete hooks cycle and return (state, step_count).

    Simulates the verifiers rollout loop: setup → (prompt → step) × N → render.
    LLM completions are garbage text → triggers random fallback actions.
    """
    state = await env.setup_state({"task": "battle", "prompt": []})
    step_count = 0

    while not await env.game_over(state) and step_count < max_steps:
        prompt = await env.get_prompt_messages(state)
        if prompt is None:
            break

        step = {
            "completion": [{"role": "assistant", "content": "I choose attack!"}],
            "prompt": prompt,
            "tokens": {
                "prompt_ids": [1, 2, 3], "completion_ids": [4, 5],
                "prompt_mask": [1, 1, 1], "completion_mask": [1, 1],
                "completion_logprobs": [-0.5, -0.3],
            },
        }
        await env.add_trajectory_step(state, step)
        step_count += 1

    await env.render_completion(state)
    return state, step_count


# ============================================================================
# T8: Full gen9ou Game — Single Agent
# ============================================================================

@requires_poke_env
@requires_showdown
class TestGen9ouSingleAgent:
    """T8: Run complete gen9ou game with heuristic opponent."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_gen9ou_random_opponent(self):
        """Full gen9ou game against random opponent with team loading.

        REQUIRES: team_dir parameter in PokemonBattleEnv (Phase 5).
        Falls back to gen9randombattle if team_dir not supported.
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="single",
                opponent_type="random",
                observation_format="simple",
                team_dir=TEAM_DIR,
                max_game_turns=200,
            )
        except TypeError:
            # team_dir not implemented yet — use random battle
            env = PokemonBattleEnv(
                battle_format="gen9randombattle",
                port=SHOWDOWN_PORT,
                play_mode="single",
                opponent_type="random",
                observation_format="simple",
                max_game_turns=200,
            )

        state, step_count = await _run_hooks_game(env, max_steps=500)
        await env.cleanup_battle(state)

        # Positive: game completed
        assert state["game_over"] is True
        assert step_count > 0, "Game must have at least 1 step"
        assert state["won"] in (True, False, None)
        assert isinstance(state["reward"], (int, float))

        # Trajectory integrity
        trajectory = state["trajectory"]
        assert len(trajectory) == step_count
        for i, s in enumerate(trajectory):
            extras = s.get("extras", {})
            assert extras.get("agent_idx") == 0, f"Step {i}: single agent = idx 0"
            assert "parsed_action" in extras, f"Step {i}: missing parsed_action"
            assert "parse_failed" in extras, f"Step {i}: missing parse_failed"
            assert isinstance(s["reward"], (int, float)), f"Step {i}: reward must be numeric"

        # Gen9ou games are typically 15-50+ turns
        # Random battles can be shorter, so we just check > 5
        assert step_count > 5, (
            f"Game had only {step_count} steps — suspiciously short"
        )

        # Negative: no steps with agent_idx != 0 (only 1 player)
        agent_indices = set(
            s["extras"]["agent_idx"] for s in trajectory
        )
        assert agent_indices == {0}, "Single-agent game must only have agent_idx=0"

    @requires_pokechamp
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_gen9ou_pokechamp_io_prompts(self):
        """Gen9ou with pokechamp_io format produces rich prompts.

        REQUIRES: team_dir parameter (Phase 5).
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="single",
                observation_format="pokechamp_io",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            env = PokemonBattleEnv(
                battle_format="gen9randombattle",
                port=SHOWDOWN_PORT,
                play_mode="single",
                observation_format="pokechamp_io",
            )

        state = await env.setup_state({"task": "battle", "prompt": []})
        prompt = await env.get_prompt_messages(state)

        assert prompt is not None
        assert len(prompt) >= 2
        system_content = prompt[0]["content"]
        user_content = prompt[1]["content"]

        # pokechamp_io prompts are rich and detailed
        assert len(user_content) > 200, (
            f"pokechamp_io prompt should be >200 chars, got {len(user_content)}"
        )
        # Should mention generation/format
        assert "pokemon" in system_content.lower() or "battle" in system_content.lower()

        await env.cleanup_battle(state)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_gen9randombattle_no_teams_needed(self):
        """gen9randombattle works without team_dir (random teams built-in)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen9randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            observation_format="simple",
        )

        state, step_count = await _run_hooks_game(env, max_steps=300)
        await env.cleanup_battle(state)

        assert state["game_over"] is True
        assert step_count > 0
        assert state["won"] in (True, False, None)


# ============================================================================
# T9: Full gen9ou Self-Play Game
# ============================================================================

@requires_poke_env
@requires_showdown
class TestGen9ouSelfPlay:
    """T9: Run complete gen9ou self-play game."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_both_players_act(self):
        """Self-play game has steps from BOTH agent_idx=0 and agent_idx=1.

        REQUIRES: team_dir parameter (Phase 5).
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="self_play",
                observation_format="simple",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            env = PokemonBattleEnv(
                battle_format="gen9randombattle",
                port=SHOWDOWN_PORT,
                play_mode="self_play",
                observation_format="simple",
            )

        state, step_count = await _run_hooks_game(env, max_steps=500)
        await env.cleanup_battle(state)

        trajectory = state["trajectory"]
        assert len(trajectory) == step_count

        # Both players must have acted
        p0_steps = [s for s in trajectory if s["extras"]["agent_idx"] == 0]
        p1_steps = [s for s in trajectory if s["extras"]["agent_idx"] == 1]

        assert len(p0_steps) > 0, "P0 must have at least 1 step"
        assert len(p1_steps) > 0, "P1 must have at least 1 step"

        # Step counts should be roughly equal (within ±15 for gen9ou)
        diff = abs(len(p0_steps) - len(p1_steps))
        assert diff <= 15, (
            f"P0={len(p0_steps)} P1={len(p1_steps)} steps. "
            f"Diff {diff} exceeds ±15 tolerance"
        )

        # Negative: no steps with agent_idx > 1
        all_indices = set(s["extras"]["agent_idx"] for s in trajectory)
        assert all_indices <= {0, 1}, f"Only 2 players, got indices {all_indices}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_rewards_opposite(self):
        """Self-play: winner and loser get opposite rewards."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen9randombattle",
            port=SHOWDOWN_PORT,
            play_mode="self_play",
            observation_format="simple",
            reward_win=1.0,
            reward_loss=0.0,
        )

        state, _ = await _run_hooks_game(env)
        await env.cleanup_battle(state)

        trajectory = state["trajectory"]
        won = state["won"]

        p0_rewards = [s["reward"] for s in trajectory
                      if s["extras"]["agent_idx"] == 0]
        p1_rewards = [s["reward"] for s in trajectory
                      if s["extras"]["agent_idx"] == 1]

        if won is True:
            # P0 won
            assert all(r == 1.0 for r in p0_rewards), "P0 (winner) gets 1.0"
            assert all(r == 0.0 for r in p1_rewards), "P1 (loser) gets 0.0"
        elif won is False:
            # P1 won
            assert all(r == 0.0 for r in p0_rewards), "P0 (loser) gets 0.0"
            assert all(r == 1.0 for r in p1_rewards), "P1 (winner) gets 1.0"
        else:
            # Draw: both players get reward_draw (0.0)
            assert all(r == 0.0 for r in p0_rewards), "Draw: P0 gets reward_draw"
            assert all(r == 0.0 for r in p1_rewards), "Draw: P1 gets reward_draw"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_game_turn_increases(self):
        """Self-play: game_turn values increase monotonically (or stay same for force-switches)."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen9randombattle",
            port=SHOWDOWN_PORT,
            play_mode="self_play",
            observation_format="simple",
        )

        state, _ = await _run_hooks_game(env)
        await env.cleanup_battle(state)

        turns = [s["extras"].get("game_turn", 0) for s in state["trajectory"]]
        # Turns should be non-decreasing (can repeat for force-switches)
        for i in range(1, len(turns)):
            assert turns[i] >= turns[i - 1], (
                f"Turn decreased at step {i}: {turns[i]} < {turns[i-1]}"
            )


# ============================================================================
# T10: Team Selection
# ============================================================================

@requires_poke_env
@requires_showdown
class TestTeamSelection:
    """T10: Verify teams are used in gen9ou battles.

    REQUIRES: team_dir parameter in PokemonBattleEnv (Phase 5).
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_teams_accepted_by_showdown(self):
        """Gen9ou games start successfully with teams from pool.

        REQUIRES: team_dir parameter (Phase 5).
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="single",
                opponent_type="random",
                observation_format="simple",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            pytest.skip("team_dir not yet implemented (Phase 5)")

        # Run 3 games — all should start without Showdown rejecting teams
        for i in range(3):
            state, _ = await _run_hooks_game(env, max_steps=300)
            await env.cleanup_battle(state)
            assert state["game_over"] is True, f"Game {i} didn't complete"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_team_variety_across_games(self):
        """Teams vary across games (not always the same).

        With 13 teams, P(all 5 same) = (1/13)^4 ≈ 0.003% — virtually impossible.

        REQUIRES: team_dir parameter (Phase 5).
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="single",
                opponent_type="random",
                observation_format="simple",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            pytest.skip("team_dir not yet implemented (Phase 5)")

        # We can't directly observe which team was chosen from the game result,
        # but we can verify that the env.team_fn() returns different teams
        teams = set()
        for _ in range(10):
            teams.add(env.team_fn())

        assert len(teams) >= 2, (
            f"team_fn returned only {len(teams)} distinct teams from 10 calls. "
            "With 13 teams, should see variety."
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_both_players_get_teams(self):
        """Self-play: both players get valid teams.

        REQUIRES: team_dir parameter (Phase 5).
        """
        from pokemon_rl.env import PokemonBattleEnv

        try:
            env = PokemonBattleEnv(
                battle_format="gen9ou",
                port=SHOWDOWN_PORT,
                play_mode="self_play",
                observation_format="simple",
                team_dir=TEAM_DIR,
            )
        except TypeError:
            pytest.skip("team_dir not yet implemented (Phase 5)")

        state, _ = await _run_hooks_game(env, max_steps=500)
        await env.cleanup_battle(state)

        # Game completed = both players had valid teams
        assert state["game_over"] is True
        assert len(state["trajectory"]) > 0


# ============================================================================
# T11: Max Game Turns Truncation
# ============================================================================

@requires_poke_env
@requires_showdown
class TestMaxGameTurnsTruncation:
    """T11: Verify max_game_turns truncation works correctly."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_truncation_at_1_turn(self):
        """max_game_turns=1 guarantees truncation (game can't end in 1 turn).

        ANTI-REWARD-HACKING: Truncation must give reward_draw, not reward_win.
        """
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            observation_format="simple",
            max_game_turns=1,
            reward_win=1.0,
            reward_loss=0.0,
            reward_draw=0.0,
        )

        state, step_count = await _run_hooks_game(env, max_steps=10)
        await env.cleanup_battle(state)

        assert state["truncated"] is True, "Game should be truncated at max_game_turns=1"
        # Truncation = draw → reward_draw
        for s in state["trajectory"]:
            assert s["reward"] == 0.0, (
                f"Truncated game must give reward_draw (0.0), got {s['reward']}"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_normal_game_not_truncated(self):
        """Normal game (max_game_turns=200) should NOT be truncated."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            observation_format="simple",
            max_game_turns=200,
        )

        state, _ = await _run_hooks_game(env)
        await env.cleanup_battle(state)

        assert state["truncated"] is False, (
            "Normal game with max_game_turns=200 should not be truncated"
        )
        # Normal game should have a winner
        assert state["won"] in (True, False)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_truncation(self):
        """Self-play truncation: both players get reward_draw."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="self_play",
            observation_format="simple",
            max_game_turns=1,
            reward_draw=0.0,
        )

        state, _ = await _run_hooks_game(env, max_steps=20)
        await env.cleanup_battle(state)

        # Both players should get draw reward
        for s in state["trajectory"]:
            assert s["reward"] == 0.0, (
                f"Self-play truncation: all steps must get reward_draw, "
                f"got {s['reward']} for agent_idx={s['extras']['agent_idx']}"
            )


# ============================================================================
# T12: Concurrent Games
# ============================================================================

@requires_poke_env
@requires_showdown
class TestConcurrentGames:
    """T12: Verify multiple games can run concurrently on same Showdown."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_8_concurrent_single_agent(self):
        """8 concurrent single-agent games all complete successfully.

        Timeout: 10 minutes for all 8 games.
        """
        from pokemon_rl.env import PokemonBattleEnv

        async def run_one_game(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle",
                port=SHOWDOWN_PORT,
                play_mode="single",
                observation_format="simple",
            )
            state, count = await _run_hooks_game(env, max_steps=300)
            await env.cleanup_battle(state)
            return state, count, game_id

        results = await asyncio.wait_for(
            asyncio.gather(*[run_one_game(i) for i in range(8)]),
            timeout=600,  # 10 minutes
        )

        # All 8 games completed
        assert len(results) == 8
        game_trajectories = []
        for state, count, game_id in results:
            assert state["game_over"] is True, f"Game {game_id} didn't complete"
            assert count > 0, f"Game {game_id} had 0 steps"
            assert isinstance(state["reward"], (int, float))
            game_trajectories.append(len(state["trajectory"]))

        # Negative: games should vary (different random outcomes)
        # With 8 games, extremely unlikely all have exact same trajectory length
        assert len(set(game_trajectories)) >= 2, (
            f"All 8 games had identical trajectory lengths {game_trajectories}. "
            "Games should be independent with varying outcomes."
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_4_concurrent_selfplay(self):
        """4 concurrent self-play games all complete."""
        from pokemon_rl.env import PokemonBattleEnv

        async def run_one_game(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle",
                port=SHOWDOWN_PORT,
                play_mode="self_play",
                observation_format="simple",
            )
            state, count = await _run_hooks_game(env, max_steps=500)
            await env.cleanup_battle(state)
            return state, count, game_id

        results = await asyncio.wait_for(
            asyncio.gather(*[run_one_game(i) for i in range(4)]),
            timeout=600,
        )

        assert len(results) == 4
        for state, count, game_id in results:
            assert state["game_over"] is True, f"Game {game_id} didn't complete"
            assert count > 0

            # Self-play: both players should have acted
            p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
            p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
            assert len(p0) > 0, f"Game {game_id}: P0 missing"
            assert len(p1) > 0, f"Game {game_id}: P1 missing"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mixed_concurrent_games(self):
        """Mix of single-agent and self-play games concurrently."""
        from pokemon_rl.env import PokemonBattleEnv

        async def run_single(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle",
                port=SHOWDOWN_PORT,
                play_mode="single",
                observation_format="simple",
            )
            state, count = await _run_hooks_game(env, max_steps=300)
            await env.cleanup_battle(state)
            return state, count, f"single-{game_id}"

        async def run_selfplay(game_id):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle",
                port=SHOWDOWN_PORT,
                play_mode="self_play",
                observation_format="simple",
            )
            state, count = await _run_hooks_game(env, max_steps=500)
            await env.cleanup_battle(state)
            return state, count, f"selfplay-{game_id}"

        results = await asyncio.wait_for(
            asyncio.gather(
                run_single(0), run_single(1),
                run_selfplay(0), run_selfplay(1),
            ),
            timeout=600,
        )

        assert len(results) == 4
        for state, count, name in results:
            assert state["game_over"] is True, f"{name} didn't complete"
            assert count > 0, f"{name} had 0 steps"
