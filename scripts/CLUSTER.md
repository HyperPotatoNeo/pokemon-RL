# Cluster Integration Guide — NERSC Perlmutter

## Quick Start

### Allocate nodes (from _CAP_tinker reservation)
```bash
# Single node:
sbatch -A m5017 --reservation=_CAP_tinker -C "gpu&hbm80g" --qos=normal \
    --time=1:00:00 --gpus-per-node=4 --nodes=1 your_script.sh

# Two nodes (for multi-node tests):
sbatch -A m5017 --reservation=_CAP_tinker -C "gpu&hbm80g" --qos=normal \
    --time=1:00:00 --gpus-per-node=4 --nodes=2 your_script.sh
```

**Reservation nodes**: nid008205, nid008268, nid008297, nid008304, nid008448, nid008480
(6 GPU nodes, hbm80g, until 2026-03-29)

### Run tests
```bash
# Unit tests (login node, no Showdown needed):
.venv/bin/python -m pytest -m unit -v

# Integration tests (login node, Showdown on localhost):
# Start Showdown first, then:
.venv/bin/python -m pytest -m integration -v

# GPU tests (compute node via sbatch):
sbatch scripts/_gpu_test.sh           # Single-node LLM tests
sbatch scripts/_gpu_test_multinode.sh  # Multi-node cross-node tests
```

## Key Findings (2026-03-21)

### Lustre flock issue with HF cache
vLLM's model loading uses `filelock` which calls `fcntl.flock()`. Lustre (`/pscratch`)
does NOT support `flock` — you get `OSError: [Errno 524]`. **Fix**: Copy model cache
to node-local `/tmp` before starting vLLM:
```bash
export HF_HOME=/tmp/hf_cache_$$
export HF_HUB_OFFLINE=1
mkdir -p $HF_HOME/hub
cp -r /pscratch/sd/s/siddart2/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507 $HF_HOME/hub/
```

### vLLM startup time
vLLM takes ~60-90s to fully start (model load + CUDA graph compilation). The API
server port (8001) only opens after compilation completes. Use `sleep 90` before
checking port availability.

### WebSocket URI format
Pokechamp's poke-env fork takes `host:port` in `ServerConfiguration` and internally
prepends `ws://` and appends `/showdown/websocket`. Do NOT pass the full URI —
`ServerConfiguration("localhost:8000", ...)` is correct. Passing
`ws://localhost:8000/showdown/websocket` would result in double-prefixing.

Note: metamon's poke-env fork may differ. Use separate venvs.

### pokechamp_io data cache paths
Pokechamp's `data_cache.py` uses hardcoded relative paths (`./poke_env/data/static/...`).
Requires a symlink at the project root: `ln -sf vendor/pokechamp/poke_env poke_env`.
Without this, pokechamp_io prompts will have empty damage calcs (0 move effects loaded).

### SSH vs srun for compute nodes
Cannot SSH to compute nodes directly (pam_slurm_adopt blocks). Use `srun` within
an allocation or `sbatch` scripts. For multi-node, use `srun --nodelist=nidXXXXXX`.

### Login node can run Showdown + integration tests
Node.js and Showdown work fine on login nodes (no GPU needed). Integration tests
(real Showdown battles with random/garbage actions) run on login nodes. Only GPU
tests (vLLM) need compute nodes.

## Architecture: Two Environments

### pokemon-rl venv (`$SCRATCH/pokemon-rl/.venv`)
- **Contains**: pokechamp, poke_env (pokechamp's fork), openai, vllm, verifiers, pytest
- **Used for**: All BattleManager tests, LLM battles via pokechamp, integration tests
- **Activate**: `source .venv/bin/activate`

### metamon venv (`$SCRATCH/metamon/.venv`)
- **Contains**: metamon, poke_env (metamon's fork), amago, gym
- **Used for**: Metamon heuristic/RL agent battles
- **Activate**: `source .venv/bin/activate && export METAMON_CACHE_DIR=...`
- **CRITICAL**: Must run from `/pscratch/sd/s/siddart2/metamon/` directory

### Why two venvs?
Pokechamp and metamon use **different forks of poke_env** with incompatible APIs:
- Pokechamp's poke_env: Has `LocalSim`, `AbyssalPlayer`, pokechamp-specific extensions
- Metamon's poke_env: Has `OpenAIGymEnv`, `SimpleHeuristicsPlayer`
They cannot coexist in the same Python process.

## Container Networking

### Single node (default)
Container uses bridge networking. Showdown on `localhost:8000` is accessible
only inside the container. All players must run in the same container.

### Multi-node (`--net=host`)
Container shares the host network. Showdown on `localhost:8000` is accessible
from other nodes via `nidXXXXXX:8000`. Required for cross-node battles.

**Strategy**: ONE Showdown server on one node. All players connect to it.
This mirrors online play — one authoritative server, multiple clients.

```
Node A (Showdown node):         Node B (player-only):
┌─────────────────────┐         ┌──────────────────┐
│ ├─ Showdown :8000   │◄────────│ ├─ Player B      │
│ ├─ Player A         │         │ └─ vLLM :8001    │
│ └─ vLLM :8001       │         └──────────────────┘
└─────────────────────┘
```

Both players use `ServerConfiguration("nidA:8000", ...)` (poke-env adds ws:// prefix).

## vLLM Setup

```bash
# On compute node (via sbatch script):
export HF_HOME=/tmp/hf_cache_$$
export HF_HUB_OFFLINE=1
mkdir -p $HF_HOME/hub
cp -r /pscratch/sd/s/siddart2/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507 $HF_HOME/hub/

.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 \
    --max-model-len 4096 \
    --no-enable-log-requests
```

**Notes**:
- MUST copy HF cache to `/tmp` (Lustre flock issue, see above)
- Use `--no-enable-log-requests` (not `--disable-log-requests`) for vllm 0.17.x
- Model must be pre-cached. Available: Qwen3-4B-Instruct-2507, Qwen3-0.6B, Qwen2.5-1.5B-Instruct
- Default `--gpu-memory-utilization 0.9` is fine for dedicated test nodes

## Metamon Agent Notes

Metamon's trained RL agents (Kakuna, Alakazam) require the AMAGO framework and
cannot be directly used as poke-env Player callbacks. They need:
1. PokeEnvWrapper (gym env interface)
2. AMAGO agent loop for inference
3. Observation/action space converters

For testing with BattleManager, use metamon's **heuristic baselines** instead:
- `EmeraldKaizo` — strong multi-layered tactical AI (1200+ lines of heuristics)
- `KaizoPlus` — risky variant
- `Gen1BossAI` — gen1-specific heuristic
- `PokeEnvHeuristic` — basic heuristic

These are poke-env Player subclasses with intelligent `choose_move()` implementations.

## Verified Test Results (2026-03-21)

| Test | Result | Details |
|------|--------|---------|
| Unit tests (207) | ALL PASS | Login node, no Showdown |
| Integration tests (28) | ALL PASS | Login node + Showdown on localhost |
| GPU: LLM vs random | PASS | Qwen3-4B, compute node, real battles |
| GPU: LLM self-play | PASS | Qwen3-4B vs itself, force-switches handled |
| GPU: Concurrent LLM | PASS | 2 concurrent LLM battles |
| GPU: Cross-node battle | PASS | Showdown on nid008205, test from nid008268 |
| GPU: Cross-node selfplay | PASS | Cross-node self-play |
