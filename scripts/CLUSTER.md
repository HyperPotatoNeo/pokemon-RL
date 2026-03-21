# Cluster Integration Guide — NERSC Perlmutter

## Quick Start

### Allocate nodes
```bash
# Single node (interactive, 4h):
salloc -A m5017 -C "gpu&hbm80g" --reservation=_CAP_tinker --qos=interactive --time=4:00:00 --gpus-per-node=4 --nodes=1 --no-shell

# Two nodes (for multi-node tests):
salloc -A m5017 -C "gpu&hbm80g" --reservation=_CAP_tinker --qos=interactive --time=4:00:00 --gpus-per-node=4 --nodes=2 --no-shell
```

### Setup node
For single-node tests (Showdown + container):
```bash
ssh nidXXXXXX "bash /pscratch/sd/s/siddart2/pokemon-rl/scripts/_setup_node.sh"
```

For multi-node (requires `--net=host` for cross-node connectivity):
```bash
# Node with Showdown:
ssh nidXXXXXX "bash /pscratch/sd/s/siddart2/pokemon-rl/scripts/_setup_node_hostnet.sh true"
# Node without Showdown:
ssh nidYYYYYY "bash /pscratch/sd/s/siddart2/pokemon-rl/scripts/_setup_node_hostnet.sh false"
```

### Run tests
```bash
# Unit tests (login node, no Showdown needed):
.venv/bin/python -m pytest -m unit -v

# Integration tests (compute node):
ssh nidXXXXXX "export HOME=... && podman-hpc exec skyrl bash -c '
cd /pscratch/sd/s/siddart2/pokemon-rl && source .venv/bin/activate
python -m pytest -m integration -v
'"
```

## Architecture: Two Environments

### pokemon-rl venv (`$SCRATCH/pokemon-rl/.venv`)
- **Contains**: pokechamp, poke_env (pokechamp's fork), openai, vllm, pytest
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
│ Container (--net=host)│         │ Container (--net=host)│
│ ├─ Showdown :8000   │◄────────│ ├─ Player B      │
│ ├─ Player A         │         │ └─ vLLM :8001    │
│ └─ vLLM :8001       │         └──────────────────┘
└─────────────────────┘
```

Both players connect to `ws://nidA:8000/showdown/websocket`.

## vLLM Setup

```bash
# Inside container (pokemon-rl venv):
source .venv/bin/activate
export HF_HOME=/pscratch/sd/s/siddart2/.cache/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.4 \
    --no-enable-log-requests
```

**Notes**:
- HF cache at `/pscratch/sd/s/siddart2/.cache/huggingface/hub/` (not `/huggingface/hub/`)
- Use `--no-enable-log-requests` (not `--disable-log-requests`) for this vllm version
- Model must be cached or downloadable. Available: Qwen3-4B-Instruct-2507, Qwen3-0.6B, Qwen2.5-1.5B-Instruct
- `--gpu-memory-utilization 0.4` leaves room for other processes on the same GPU

## WebSocket URI Formats

Two different poke_env forks use different URI formats:

| Fork | ServerConfiguration format | Example |
|------|--------------------------|---------|
| Pokechamp's poke_env | `"localhost:8000"` | `ServerConfiguration("localhost:8000", "https://...")` |
| Metamon's poke_env | `"ws://localhost:8000/showdown/websocket"` | `ServerConfiguration("ws://localhost:8000/showdown/websocket", "https://...")` |

For cross-node, replace `localhost` with the node hostname (e.g., `nid008268`).

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

## Verified Test Results (2026-03-20)

| Test | Result | Details |
|------|--------|---------|
| EmeraldKaizo vs EmeraldKaizo (gen1randombattle) | PASS | 43 decisions, 0 defaults |
| Qwen3-4B vs Random (BattleManager turn-by-turn) | PASS | 33 decisions, reasoning traces captured |
| Qwen3-4B self-play (BattleManager) | PASS | 60 decisions, force-switches handled |
| Multi-node EmeraldKaizo (nid008268 vs nid008205) | PASS | 37 decisions, 0 defaults, `--net=host` |
