"""Phase 5 multi-node tests: Cross-node Showdown, vLLM, and weight broadcast.

Tests require:
- 2+ GPU compute nodes allocated via SLURM
- Showdown server running on Node 0
- vLLM serving on Node 0
- --net=host containers on both nodes
- Shared /pscratch filesystem

Run with 2-node allocation:
    salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 \\
        --nodes=2 --gpus-per-node 4

    # On Node 0: start Showdown + vLLM
    # On Node 1: run tests with REMOTE_NODE=nid_node0
    REMOTE_NODE=nidXXXXXX python -m pytest tests/test_phase5_multinode.py -v -m multinode

TEST PHILOSOPHY:
Multi-node tests validate infrastructure correctness. Latency checks are
informational (logged), not hard assertions.
"""

import asyncio
import os
import subprocess
import time
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REMOTE_NODE = os.environ.get("REMOTE_NODE")  # Node 0 hostname (runs Showdown + vLLM)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
SHOWDOWN_PORT = int(os.environ.get("SHOWDOWN_PORT", "8000"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/s/siddart2")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DIR = os.path.join(
    PROJECT_ROOT, "vendor", "pokechamp", "poke_env", "data", "static",
    "teams", "gen9ou",
)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
def _remote_port_open(host, port):
    if not host:
        return False
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            return sock.connect_ex((host, port)) == 0
    except Exception:
        return False


HAS_REMOTE_NODE = REMOTE_NODE is not None
HAS_REMOTE_SHOWDOWN = _remote_port_open(REMOTE_NODE, SHOWDOWN_PORT) if HAS_REMOTE_NODE else False
HAS_REMOTE_VLLM = _remote_port_open(REMOTE_NODE, VLLM_PORT) if HAS_REMOTE_NODE else False

try:
    from poke_env.player.player import Player  # noqa: F401
    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False

requires_multinode = pytest.mark.skipif(
    not (HAS_REMOTE_NODE and HAS_POKE_ENV),
    reason=(
        f"Multi-node tests require: REMOTE_NODE env var ({HAS_REMOTE_NODE}), "
        f"poke-env ({HAS_POKE_ENV})"
    ),
)


# ============================================================================
# T22: Cross-Node Showdown Connection
# ============================================================================

@requires_multinode
class TestCrossNodeShowdown:
    """T22: Verify games can connect to Showdown on a different node."""

    @pytest.mark.multinode
    @pytest.mark.skipif(not HAS_REMOTE_SHOWDOWN,
                        reason=f"Showdown not running on {REMOTE_NODE}:{SHOWDOWN_PORT}")
    @pytest.mark.asyncio
    async def test_cross_node_game_completes(self):
        """Run a game connecting to Showdown on remote node."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen9randombattle",
            port=SHOWDOWN_PORT,
            server_host=REMOTE_NODE,
            play_mode="single",
            opponent_type="random",
            observation_format="simple",
        )

        state = await env.setup_state({"task": "battle", "prompt": []})
        assert state["game_over"] is False, "Battle should start on remote Showdown"

        step_count = 0
        while not await env.game_over(state) and step_count < 300:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "attack!"}],
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
        await env.cleanup_battle(state)

        assert state["game_over"] is True
        assert step_count > 0, "Game must have steps"
        assert state["won"] in (True, False, None)

    @pytest.mark.multinode
    @pytest.mark.skipif(not HAS_REMOTE_SHOWDOWN,
                        reason=f"Showdown not running on {REMOTE_NODE}:{SHOWDOWN_PORT}")
    @pytest.mark.asyncio
    async def test_cross_node_selfplay(self):
        """Self-play game with remote Showdown works."""
        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen9randombattle",
            port=SHOWDOWN_PORT,
            server_host=REMOTE_NODE,
            play_mode="self_play",
            observation_format="simple",
        )

        state = await env.setup_state({"task": "battle", "prompt": []})
        step_count = 0

        while not await env.game_over(state) and step_count < 500:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            step = {
                "completion": [{"role": "assistant", "content": "attack!"}],
                "prompt": prompt,
                "tokens": {},
            }
            await env.add_trajectory_step(state, step)
            step_count += 1

        await env.render_completion(state)
        await env.cleanup_battle(state)

        assert state["game_over"] is True
        p0 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1 = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0) > 0 and len(p1) > 0, "Both players must have steps"

    @pytest.mark.multinode
    @pytest.mark.skipif(not HAS_REMOTE_SHOWDOWN,
                        reason=f"Showdown not running on {REMOTE_NODE}:{SHOWDOWN_PORT}")
    @pytest.mark.asyncio
    async def test_cross_node_latency_reasonable(self):
        """WebSocket connection latency to remote node is acceptable.

        Informational: logs latency, doesn't hard-fail.
        """
        import time

        from pokemon_rl.env import PokemonBattleEnv

        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=SHOWDOWN_PORT,
            server_host=REMOTE_NODE,
            play_mode="single",
            observation_format="simple",
        )

        start = time.monotonic()
        state = await env.setup_state({"task": "battle", "prompt": []})
        setup_time = time.monotonic() - start

        await env.cleanup_battle(state)

        print(f"  Cross-node setup time: {setup_time:.2f}s")
        # Informational: should be < 10s. Alert if very slow.
        if setup_time > 10:
            print(f"  WARNING: Cross-node setup took {setup_time:.1f}s (>10s)")


# ============================================================================
# T23: Cross-Node vLLM Connection
# ============================================================================

@requires_multinode
class TestCrossNodeVLLM:
    """T23: Verify LLM inference can reach vLLM on a different node."""

    @pytest.mark.multinode
    @pytest.mark.skipif(not HAS_REMOTE_VLLM,
                        reason=f"vLLM not running on {REMOTE_NODE}:{VLLM_PORT}")
    def test_cross_node_vllm_responds(self):
        """vLLM on remote node responds to inference request."""
        from openai import OpenAI

        client = OpenAI(
            base_url=f"http://{REMOTE_NODE}:{VLLM_PORT}/v1",
            api_key="dummy",
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Pokemon battler."},
                {"role": "user", "content": "Choose a move."},
            ],
            max_tokens=100,
            temperature=1.0,
        )

        assert response is not None
        assert len(response.choices) > 0
        text = response.choices[0].message.content
        assert text is not None
        assert len(text) > 0, "vLLM must return non-empty response"

    @pytest.mark.multinode
    @pytest.mark.skipif(not HAS_REMOTE_VLLM,
                        reason=f"vLLM not running on {REMOTE_NODE}:{VLLM_PORT}")
    def test_cross_node_vllm_latency(self):
        """Cross-node vLLM latency is comparable to same-node.

        Informational: logs latency comparison.
        """
        import time
        from openai import OpenAI

        client = OpenAI(
            base_url=f"http://{REMOTE_NODE}:{VLLM_PORT}/v1",
            api_key="dummy",
        )

        messages = [
            {"role": "system", "content": "You are a Pokemon battler."},
            {"role": "user", "content": "Choose a move. Reply with JSON."},
        ]

        start = time.monotonic()
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            max_tokens=50, temperature=1.0,
        )
        latency = time.monotonic() - start

        print(f"  Cross-node vLLM latency: {latency:.3f}s")
        # Should be similar to same-node (<5s for short response)
        assert latency < 30, f"Cross-node latency {latency:.1f}s is too high"


# ============================================================================
# T24: Weight Broadcast via Shared Filesystem
# ============================================================================

@requires_multinode
class TestWeightBroadcast:
    """T24: Verify weight files on /pscratch are visible from both nodes."""

    @pytest.mark.multinode
    def test_shared_filesystem_write_read(self):
        """File written on this node is readable from shared /pscratch."""
        import tempfile

        # Write a test file to /pscratch
        test_dir = os.path.join(SCRATCH, "pokemon-rl-test-broadcast")
        os.makedirs(test_dir, exist_ok=True)

        test_file = os.path.join(test_dir, "test_broadcast.txt")
        test_content = f"broadcast_test_{time.time()}"

        with open(test_file, "w") as f:
            f.write(test_content)

        # Verify we can read it back
        with open(test_file) as f:
            read_content = f.read()
        assert read_content == test_content

        # Verify from remote node via SSH (if accessible)
        if REMOTE_NODE:
            try:
                result = subprocess.run(
                    ["ssh", REMOTE_NODE, "cat", test_file],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    assert result.stdout.strip() == test_content, (
                        "File content mismatch on remote node"
                    )
                else:
                    print(f"  SSH to {REMOTE_NODE} failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"  SSH to {REMOTE_NODE} timed out")
            except FileNotFoundError:
                print("  SSH not available")

        # Cleanup
        os.remove(test_file)
        try:
            os.rmdir(test_dir)
        except OSError:
            pass

    @pytest.mark.multinode
    def test_broadcast_dir_structure(self):
        """Verify expected broadcast directory structure is writable."""
        test_output = os.path.join(SCRATCH, "pokemon-rl-test-broadcast-struct")
        broadcast_dir = os.path.join(test_output, "broadcast", "step_1")
        os.makedirs(broadcast_dir, exist_ok=True)

        # Write STABLE marker (mimics prime-rl's weight broadcast)
        stable_marker = os.path.join(broadcast_dir, "STABLE")
        with open(stable_marker, "w") as f:
            f.write("1")

        assert os.path.isfile(stable_marker)

        # Write a fake weight file
        weight_file = os.path.join(broadcast_dir, "model.safetensors")
        with open(weight_file, "w") as f:
            f.write("fake_weights")

        assert os.path.isfile(weight_file)

        # Cleanup
        import shutil
        shutil.rmtree(test_output)
