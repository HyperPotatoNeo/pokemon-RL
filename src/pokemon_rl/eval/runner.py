"""Main eval runner — loops over opponents and runs standard verifiers eval.

Usage:
    python -m pokemon_rl.eval.runner configs/pokemon/eval_example.toml
    python -m pokemon_rl.eval.runner configs/pokemon/eval_example.toml --node_rank 0 --n_nodes 2
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path

logger = logging.getLogger(__name__)


async def _generate_rollout(
    semaphore: asyncio.Semaphore,
    client: "AsyncOpenAI",  # noqa: F821
    env: "vf.Environment",  # noqa: F821
    model_name: str,
    example: dict,
    sampling_args: dict,
) -> dict:
    """Generate and score a single rollout. Standalone — no prime_rl dependency."""
    import verifiers as vf

    rollout_input = vf.RolloutInput(**example)
    state = await env.run_rollout(rollout_input, client, model_name, sampling_args, semaphore)
    await env.rubric.score_rollout(state, score_sem=semaphore)
    return state


def start_vllm_server(
    model_name: str,
    base_url: str,
    gpu_ids: list[int],
) -> subprocess.Popen:
    """Start a vLLM server as a subprocess.

    Args:
        model_name: HuggingFace model ID
        base_url: Expected base_url (used to extract port)
        gpu_ids: GPU IDs to use (via CUDA_VISIBLE_DEVICES)

    Returns:
        subprocess.Popen handle
    """
    import re

    port_match = re.search(r":(\d+)", base_url)
    port = int(port_match.group(1)) if port_match else 8002

    env_vars = dict(__import__("os").environ)
    env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--trust-remote-code",
        "--data-parallel-size", str(len(gpu_ids)),
    ]

    logger.info(f"Starting opponent vLLM: {model_name} on GPUs {gpu_ids} port {port}")
    proc = subprocess.Popen(
        cmd,
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


async def wait_for_health(base_url: str, timeout: float = 300, interval: float = 5) -> None:
    """Wait for a vLLM server to become healthy."""
    import socket
    import re

    host_match = re.search(r"//([^:]+):(\d+)", base_url)
    if not host_match:
        raise ValueError(f"Cannot parse host:port from base_url: {base_url}")
    host = host_match.group(1)
    port = int(host_match.group(2))

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                if sock.connect_ex((host, port)) == 0:
                    logger.info(f"vLLM server ready at {base_url}")
                    return
        except Exception:
            pass
        await asyncio.sleep(interval)

    raise TimeoutError(f"vLLM server at {base_url} not ready after {timeout}s")


async def run_pokemon_eval(config: "PokemonEvalConfig") -> dict:  # noqa: F821
    """Main eval entry point.

    Args:
        config: Parsed PokemonEvalConfig

    Returns:
        Dict of {opponent_name: stats_dict}
    """
    from openai import AsyncOpenAI

    from pokemon_rl.env import PokemonBattleEnv
    from pokemon_rl.eval.config import compute_node_share
    from pokemon_rl.eval.report import compute_stats, generate_summary, save_results

    clients = [AsyncOpenAI(base_url=config.agent_base_url, api_key="EMPTY")]
    all_results: dict[str, dict] = {}

    battles_this_node = compute_node_share(
        config.n_battles_per_opp, config.node_rank, config.n_nodes
    )

    if battles_this_node == 0:
        logger.info(f"Node {config.node_rank}: no battles to run (n_battles < n_nodes)")
        return all_results

    logger.info(
        f"Node {config.node_rank}/{config.n_nodes}: "
        f"running {battles_this_node} battles per opponent "
        f"({len(config.opponents)} opponents)"
    )

    for opp in config.opponents:
        opp_type = opp.opponent_type_for_env
        logger.info(f"--- Evaluating vs {opp.name} (type={opp.type}, opp_type={opp_type}) ---")

        env_kwargs: dict = {
            "battle_format": config.battle_format,
            "port": config.showdown_port,
            "play_mode": "single",
            "opponent_type": opp_type,
            "num_battles": battles_this_node,
            "max_concurrent_battles": config.max_concurrent_battles,
            "max_game_turns": config.max_game_turns,
            "observation_format": config.observation_format,
        }
        if config.team_dir:
            env_kwargs["team_dir"] = config.team_dir
        if opp.type == "llm":
            env_kwargs["llm_opponent_kwargs"] = {
                "base_url": opp.base_url,
                "model_name": opp.model_name,
                "max_tokens": opp.max_tokens,
                "temperature": opp.temperature,
                "observation_format": opp.observation_format,
            }

        # Manage opponent server lifecycle
        opp_proc = None
        if opp.type == "llm" and opp.gpu_ids:
            opp_proc = start_vllm_server(opp.model_name, opp.base_url, opp.gpu_ids)
            await wait_for_health(opp.base_url)

        try:
            env = PokemonBattleEnv(**env_kwargs)
            dataset = env.get_eval_dataset(n=battles_this_node)

            semaphore = asyncio.Semaphore(config.max_concurrent_battles)
            sampling_args = {
                "max_tokens": config.sampling_max_tokens,
                "temperature": config.sampling_temperature,
                "extra_body": {},
            }

            examples = dataset.to_list()
            logger.info(f"Running {len(examples)} battles vs {opp.name}...")

            states = await asyncio.gather(*[
                _generate_rollout(
                    semaphore, client, env, config.agent_model,
                    example, sampling_args,
                )
                for client, example in zip(cycle(clients), examples)
            ])

            results_path = save_results(states, opp.name, config.output_dir, config.node_rank)
            stats = compute_stats(states)
            all_results[opp.name] = stats

            logger.info(
                f"vs {opp.name}: "
                f"wins={stats['wins']} losses={stats['losses']} draws={stats['draws']} "
                f"win_rate={stats['win_rate']:.1%} +/- {stats['win_rate_stderr']:.1%} "
                f"(saved to {results_path})"
            )
        finally:
            if opp_proc:
                logger.info(f"Stopping opponent vLLM (PID {opp_proc.pid})")
                opp_proc.terminate()
                try:
                    opp_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    opp_proc.kill()

    summary = generate_summary(all_results, config.output_dir, config.node_rank)
    print("\n=== Eval Summary ===")
    print(summary)

    return all_results


def main():
    """CLI entry point: python -m pokemon_rl.eval.runner <config.toml> [--flags]"""
    from pokemon_rl.eval.config import PokemonEvalConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m pokemon_rl.eval.runner <config.toml> [--node_rank N] [--n_nodes N]")
        sys.exit(1)

    config_path = sys.argv[1]

    # Parse optional CLI overrides
    config = PokemonEvalConfig.from_toml(config_path)

    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        if argv[i] == "--node_rank" and i + 1 < len(argv):
            config.node_rank = int(argv[i + 1])
            i += 2
        elif argv[i] == "--n_nodes" and i + 1 < len(argv):
            config.n_nodes = int(argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {argv[i]}")
            sys.exit(1)

    asyncio.run(run_pokemon_eval(config))


if __name__ == "__main__":
    main()
