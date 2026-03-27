#!/usr/bin/env python3
"""Inspect interleaved trajectory conversations.

Runs games against abyssal via the full PokemonBattleEnv + vLLM pipeline
and prints detailed conversation breakdowns for manual inspection.

Usage (inside container with venv + Showdown + vLLM running):
    python scripts/inspect_interleaved.py [--games N] [--port 8001] [--model MODEL]

Requires:
    - Showdown running on port 8000
    - vLLM running on the specified port (default 8001)
"""

import argparse
import asyncio
import json
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Inspect interleaved trajectories")
    p.add_argument("--games", type=int, default=5, help="Number of games to run")
    p.add_argument("--port", type=int, default=8001, help="vLLM port")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--reasoning-tokens", type=int, default=512)
    p.add_argument("--extraction-tokens", type=int, default=50)
    return p.parse_args()


async def run_game(env, client, model, sampling_args, game_idx):
    """Run a single interleaved game and return (state, elapsed)."""
    import verifiers as vf

    example = {"prompt": [{"role": "user", "content": "battle"}], "example_id": game_idx}
    inp = vf.RolloutInput(**example)

    t0 = time.perf_counter()
    state = await env.rollout(inp, client, model, sampling_args)
    elapsed = time.perf_counter() - t0
    return state, elapsed


def print_game_report(state, game_idx, elapsed):
    """Print detailed report for one game."""
    trajectory = state.get("trajectory", [])
    won = state.get("won")
    reward = state.get("reward", 0.0)

    outcome = "WIN" if won is True else "LOSS" if won is False else "DRAW"
    print(f"\n{'='*80}")
    print(f"GAME {game_idx + 1}: {outcome} (reward={reward:.1f}, {elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"  Trajectory steps: {len(trajectory)}")
    print(f"  Game turns: {state.get('game_turn', 0)}")

    # Count phases
    reasoning_steps = [s for s in trajectory if s.get("extras", {}).get("phase") == 0]
    extraction_steps = [s for s in trajectory if s.get("extras", {}).get("phase") == 1]
    parse_failures = sum(1 for s in trajectory if s.get("extras", {}).get("parse_failed"))

    print(f"  Reasoning steps: {len(reasoning_steps)}")
    print(f"  Extraction steps: {len(extraction_steps)}")
    print(f"  Parse failures: {parse_failures}")

    # Token counts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for step in trajectory:
        tokens = step.get("tokens")
        if tokens:
            total_prompt_tokens = max(total_prompt_tokens, len(tokens.get("prompt_ids", [])))
            total_completion_tokens += len(tokens.get("completion_ids", []))

    print(f"  Max prompt tokens: {total_prompt_tokens}")
    print(f"  Total completion tokens: {total_completion_tokens}")
    print(f"  Estimated total tokens: {total_prompt_tokens + total_completion_tokens}")

    # Print conversation
    print(f"\n--- CONVERSATION ---")
    for i, step in enumerate(trajectory):
        extras = step.get("extras", {})
        phase = extras.get("phase", "?")
        game_step = extras.get("game_step", "?")
        phase_name = "REASONING" if phase == 0 else "EXTRACTION"

        # Prompt info
        prompt = step.get("prompt", [])
        if isinstance(prompt, list) and prompt:
            last_user = None
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            if last_user:
                # Truncate for display
                display = last_user[:200] + "..." if len(last_user) > 200 else last_user
                print(f"\n  [Step {i}] Turn {game_step}, {phase_name}")
                print(f"  User ({len(last_user)} chars): {display}")

        # Completion
        completion = step.get("completion", "")
        if isinstance(completion, list):
            for msg in completion:
                if isinstance(msg, dict):
                    completion = msg.get("content", "")
                    break
        if isinstance(completion, str):
            display = completion[:300] + "..." if len(completion) > 300 else completion
            print(f"  Assistant: {display}")

        # Token info per step
        tokens = step.get("tokens")
        if tokens:
            prompt_len = len(tokens.get("prompt_ids", []))
            comp_len = len(tokens.get("completion_ids", []))
            truncated = tokens.get("is_truncated", False)
            trunc_marker = " [TRUNCATED]" if truncated else ""
            print(f"  Tokens: prompt={prompt_len}, completion={comp_len}{trunc_marker}")

        if extras.get("parse_failed"):
            print(f"  ** PARSE FAILED — used fallback action")
        if extras.get("parsed_action"):
            print(f"  Action: {extras['parsed_action']}")

    # First turn vs subsequent turn size comparison
    if len(trajectory) >= 4:
        first_prompt = trajectory[0].get("prompt", [])
        third_prompt = trajectory[2].get("prompt", []) if len(trajectory) > 2 else []
        first_len = sum(len(m.get("content", "")) for m in first_prompt if isinstance(m, dict))
        # For subsequent, find a reasoning step after turn 0
        for step in trajectory[2:]:
            if step.get("extras", {}).get("phase") == 0:
                sp = step.get("prompt", [])
                if isinstance(sp, list) and sp:
                    last_user = None
                    for msg in reversed(sp):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            last_user = msg.get("content", "")
                            break
                    if last_user:
                        print(f"\n  First-turn user prompt: {first_len} chars")
                        print(f"  Subsequent-turn user prompt: {len(last_user)} chars")
                        if first_len > 0:
                            print(f"  Reduction: {(1 - len(last_user)/first_len)*100:.0f}%")
                break


async def main():
    args = parse_args()

    # Lazy imports to fail fast on missing deps
    from openai import AsyncOpenAI
    from pokemon_rl.env import PokemonBattleEnv

    env = PokemonBattleEnv(
        battle_format="gen9ou",
        play_mode="single",
        opponent_type="abyssal",
        port=8000,
        observation_format="full_obs_cot",
        reward_win=1.0,
        reward_loss=0.0,
        reward_draw=0.0,
        max_game_turns=200,
        num_battles=args.games,
        team_dir="vendor/pokechamp/poke_env/data/static/teams/gen9ou",
        max_concurrent_battles=4,
        interleaved=True,
        reasoning_tokens=args.reasoning_tokens,
        extraction_tokens=args.extraction_tokens,
    )

    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="dummy")

    sampling_args = {"temperature": 1.0, "max_tokens": args.reasoning_tokens}

    # Summary stats
    wins, losses, draws = 0, 0, 0
    total_steps = 0
    total_tokens = 0
    total_time = 0.0

    for i in range(args.games):
        try:
            state, elapsed = await run_game(env, client, args.model, sampling_args, i)
            print_game_report(state, i, elapsed)

            won = state.get("won")
            if won is True:
                wins += 1
            elif won is False:
                losses += 1
            else:
                draws += 1

            traj = state.get("trajectory", [])
            total_steps += len(traj)
            total_time += elapsed

            # Max tokens in this game
            for step in traj:
                tokens = step.get("tokens")
                if tokens:
                    total_tokens = max(total_tokens, len(tokens.get("prompt_ids", [])) + len(tokens.get("completion_ids", [])))

        except Exception as e:
            print(f"\nGAME {i + 1}: ERROR — {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.games} games")
    print(f"{'='*80}")
    print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"  Total trajectory steps: {total_steps}")
    print(f"  Max tokens in any game: {total_tokens}")
    print(f"  Total time: {total_time:.1f}s ({total_time/max(args.games,1):.1f}s/game)")

    # Compare to branching estimate
    avg_steps_per_game = total_steps / max(args.games, 1)
    branching_tokens_est = avg_steps_per_game / 2 * 3400  # ~3400 tokens per turn in branching
    print(f"  Branching estimate (per game): {branching_tokens_est:.0f} tokens")
    print(f"  Interleaved actual (max game): {total_tokens} tokens")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
