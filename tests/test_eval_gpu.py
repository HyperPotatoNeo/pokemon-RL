"""GPU tests for pokemon-rl eval feature.

Requires: compute node + Showdown + vLLM serving agent model.
Optionally: opponent vLLM for LLM-vs-LLM tests.

Environment variables:
    VLLM_HOST       — Agent vLLM host (default: localhost)
    VLLM_PORT       — Agent vLLM port (default: 8001)
    MODEL_NAME      — Agent model name (default: Qwen/Qwen3-4B-Instruct-2507)
    SHOWDOWN_PORT   — Showdown port (default: 8000)
    OPP_VLLM_PORT   — Opponent vLLM port (default: 8002)
    OPP_MODEL_NAME  — Opponent model name (default: Qwen/Qwen2.5-1.5B-Instruct)
"""

from __future__ import annotations

import os
import socket

import pytest

# --- Configuration ---
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
SHOWDOWN_PORT = int(os.environ.get("SHOWDOWN_PORT", "8000"))
OPP_VLLM_PORT = int(os.environ.get("OPP_VLLM_PORT", "8002"))
OPP_MODEL_NAME = os.environ.get("OPP_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")


# --- Capability detection ---
def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


HAS_VLLM = _is_port_open(VLLM_HOST, VLLM_PORT)
HAS_SHOWDOWN = _is_port_open("localhost", SHOWDOWN_PORT)
HAS_OPP_VLLM = _is_port_open(VLLM_HOST, OPP_VLLM_PORT)


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


requires_vllm = pytest.mark.skipif(
    not HAS_VLLM, reason=f"vLLM not running at {VLLM_HOST}:{VLLM_PORT}"
)
requires_showdown = pytest.mark.skipif(
    not HAS_SHOWDOWN, reason=f"Showdown not running on port {SHOWDOWN_PORT}"
)
requires_poke_env = pytest.mark.skipif(not _has_poke_env(), reason="poke-env not installed")
requires_verifiers = pytest.mark.skipif(not _has_verifiers(), reason="verifiers not installed")
requires_opp_vllm = pytest.mark.skipif(
    not HAS_OPP_VLLM, reason=f"Opponent vLLM not running at {VLLM_HOST}:{OPP_VLLM_PORT}"
)


@requires_poke_env
@requires_showdown
@requires_vllm
@requires_verifiers
class TestEvalGPUvsHeuristic:
    """GPU tests: real agent LLM vs heuristic opponents."""

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_eval_agent_vs_random_real_moves(self):
        """Agent vLLM vs random: agent produces mostly valid moves."""
        from openai import AsyncOpenAI

        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="random",
            num_battles=5,
            max_concurrent_battles=4,
            max_game_turns=100,
            observation_format="simple",
        )

        client = AsyncOpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="EMPTY",
        )

        import verifiers as vf

        dataset = env.get_eval_dataset(n=5)
        examples = dataset.to_list()

        import asyncio

        sem = asyncio.Semaphore(4)
        states = []
        for ex in examples:
            ri = vf.RolloutInput(**ex)
            state = await env.run_rollout(
                ri, client, MODEL_NAME,
                {"max_tokens": 256, "temperature": 0.7, "extra_body": {}},
                sem,
            )
            await env.rubric.score_rollout(state, score_sem=sem)
            states.append(state)

        assert len(states) == 5

        # Check that most battles completed with real moves
        total_parse_failures = sum(
            s.get("metrics", {}).get("parse_failures", 0) for s in states
        )
        total_turns = sum(
            s.get("metrics", {}).get("game_turns", 0) for s in states
        )

        # Real model should parse at least half the moves
        if total_turns > 0:
            failure_rate = total_parse_failures / total_turns
            assert failure_rate < 0.5, (
                f"Parse failure rate {failure_rate:.0%} too high — "
                f"agent model may not be generating valid moves"
            )

        # qwen3-4b should beat random at least once
        wins = sum(1 for s in states if s.get("metrics", {}).get("won") == 1)
        assert wins >= 1, "Agent should win at least 1 of 5 games vs random"

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_eval_agent_vs_abyssal_completes(self):
        """Agent vLLM vs abyssal: battles complete without hang."""
        from openai import AsyncOpenAI

        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="abyssal",
            num_battles=3,
            max_concurrent_battles=2,
            max_game_turns=100,
            observation_format="simple",
        )

        client = AsyncOpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="EMPTY",
        )

        import verifiers as vf

        dataset = env.get_eval_dataset(n=3)
        examples = dataset.to_list()

        import asyncio

        sem = asyncio.Semaphore(2)
        states = []
        for ex in examples:
            ri = vf.RolloutInput(**ex)
            state = await env.run_rollout(
                ri, client, MODEL_NAME,
                {"max_tokens": 256, "temperature": 0.7, "extra_body": {}},
                sem,
            )
            await env.rubric.score_rollout(state, score_sem=sem)
            states.append(state)

        assert len(states) == 3
        for i, s in enumerate(states):
            turns = s.get("metrics", {}).get("game_turns", 0)
            assert turns > 0, f"Battle {i}: should have turns"


@requires_poke_env
@requires_showdown
@requires_vllm
@requires_verifiers
@requires_opp_vllm
class TestEvalGPUvsLLM:
    """GPU tests: real agent LLM vs LLM opponent."""

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_eval_llm_vs_llm_completes(self):
        """Agent vLLM vs opponent vLLM: battles complete with real moves."""
        from openai import AsyncOpenAI

        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            play_mode="single",
            opponent_type="llm",
            num_battles=3,
            max_concurrent_battles=2,
            max_game_turns=100,
            observation_format="simple",
            llm_opponent_kwargs={
                "base_url": f"http://{VLLM_HOST}:{OPP_VLLM_PORT}/v1",
                "model_name": OPP_MODEL_NAME,
                "max_tokens": 256,
                "temperature": 0.7,
                "observation_format": "simple",
            },
        )

        client = AsyncOpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="EMPTY",
        )

        import verifiers as vf

        dataset = env.get_eval_dataset(n=3)
        examples = dataset.to_list()

        import asyncio

        sem = asyncio.Semaphore(2)
        states = []
        for ex in examples:
            ri = vf.RolloutInput(**ex)
            state = await env.run_rollout(
                ri, client, MODEL_NAME,
                {"max_tokens": 256, "temperature": 0.7, "extra_body": {}},
                sem,
            )
            await env.rubric.score_rollout(state, score_sem=sem)
            states.append(state)

        assert len(states) == 3

        # Both sides should produce some real moves
        for i, s in enumerate(states):
            turns = s.get("metrics", {}).get("game_turns", 0)
            assert turns > 0, f"Battle {i}: should have turns (LLM vs LLM)"
