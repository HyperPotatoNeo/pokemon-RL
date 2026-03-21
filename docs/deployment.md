# Deployment on NERSC Perlmutter

pokemon-rl runs on NERSC's Perlmutter supercomputer. Login nodes can run unit tests; GPU compute nodes run integration tests and training.

## Environment Overview

```
Login node:  256 CPUs, no GPUs, no podman-hpc, no Showdown
             Can: edit files, run unit tests, use git
             Cannot: run poke-env battles, use GPUs

Compute node: 4x A100-80GB, podman-hpc containers, Showdown server
              Can: everything
              Requires: salloc allocation
```

## Setup Steps

### 1. Allocate a Compute Node

```bash
# From the _CAP_tinker reservation (if available):
salloc -A m5017_g --reservation=_CAP_tinker -C "gpu&hbm80g" \
  --qos=interactive --time 4:00:00 --gpus-per-node 4 -N 1 --no-shell

# Without reservation:
salloc -A m5017 -C "gpu&hbm80g" --qos=interactive \
  --time 4:00:00 --gpus-per-node 4 -N 1 --no-shell
```

Check allocated node: `squeue --me`

### 2. Run Setup Script

```bash
# SSH to the node and run setup:
ssh nid008268 "export HOME=/pscratch/sd/s/siddart2 && \
  export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && \
  bash /pscratch/sd/s/siddart2/pokemon-rl/scripts/setup_node.sh"
```

`setup_node.sh` does:
1. Starts `skyrl` container (detached, with GPU passthrough)
2. Starts Showdown server inside container (port 8000)
3. Creates `.venv`, installs pokechamp + pokemon-rl

### 3. Run Tests

```bash
bash scripts/run_tests_remote.sh nid008268 -v
```

## Container Details

Image: `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`
- Python 3.12, CUDA 12.8, Ray 2.51.1, `uv` package manager
- No pip/conda — use `uv pip install`

```bash
# Container launch (done by setup_node.sh):
podman-hpc run --rm -d \
  --user "$(id -u):$(id -g)" --replace --name skyrl \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME \
  -v "$HOME":"$HOME" -w "$HOME/pokemon-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  sleep infinity
```

Key flags:
- `--gpu --nccl` — GPU passthrough with NCCL support
- `--userns keep-id` — UID mapping (files on Lustre accessible)
- `--shm-size=8g` — Shared memory for IPC
- `-v "$HOME":"$HOME"` — Mount scratch filesystem

### Running Commands Inside Container

```bash
# Interactive:
podman-hpc exec -it skyrl /bin/bash

# Non-interactive (from login node via SSH):
ssh nid008268 "export HOME=/pscratch/sd/s/siddart2 && \
  export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && \
  podman-hpc exec skyrl bash /path/to/script.sh"
```

**Nested quoting breaks.** Always write commands to a script file first, then execute the script.

## Package Installation

```bash
# Inside container:
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
cd /pscratch/sd/s/siddart2/pokemon-rl
source .venv/bin/activate

uv pip install -e /pscratch/sd/s/siddart2/pokechamp  # poke-env + deps
uv pip install -e ".[test]"                           # pokemon-rl + pytest
```

## Showdown Server

Pokemon Showdown is a Node.js application. Node.js binary is installed at:
`/pscratch/sd/s/siddart2/node-v20.18.1-linux-x64/bin/node`

Showdown directory: `/pscratch/sd/s/siddart2/pokechamp/pokemon-showdown`

```bash
# Start manually (inside container):
export PATH=/pscratch/sd/s/siddart2/node-v20.18.1-linux-x64/bin:$PATH
cd /pscratch/sd/s/siddart2/pokechamp/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

`ShowdownEngine` in `engine.py` can also manage the process, but `setup_node.sh` starts it separately so it persists across test runs.

## Multi-Node Battles

For cross-node play (e.g., 2 GPU nodes running different models):

### Requirements

1. **`--net=host` on all containers.** Default bridge networking isolates the container — players cannot reach a Showdown server on another node. Use `scripts/_setup_node_hostnet.sh` instead of `setup_node.sh`:
   ```bash
   ssh nidXXXXXX "export HOME=$SCRATCH && \
     export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman && \
     bash $SCRATCH/pokemon-rl/scripts/_setup_node_hostnet.sh true"
   ```

2. **One Showdown server, all players connect to it.** Run Showdown on one node, set `server_host` on others:
   ```python
   BattleManager(server_host="nid008268", port=8000)
   ```

3. **Separate processes per node.** poke-env's `_battle_against` runs both players in the same process. Cross-node, this hangs because the challenge/accept matchmaking flow times out when both WebSocket connections go over the network from the same event loop. For real cross-node play, run one player process per node, each connecting to the shared Showdown server independently.

### GPU Tests (vLLM)

The GPU tests (`tests/test_phase4_gpu.py`) need vLLM serving a model:

```bash
# Inside container on the GPU node:
source .venv/bin/activate
export HF_HOME=/pscratch/sd/s/siddart2/.cache/huggingface
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 --max-model-len 4096 \
    --no-enable-log-requests &

# Run GPU tests:
VLLM_HOST=localhost VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
SHOWDOWN_PORT=8000 python -m pytest tests/test_phase4_gpu.py -v
```

## Filesystem Notes

- **$SCRATCH** (`/pscratch/sd/s/siddart2/`): Fast Lustre, purged periodically. All code and data here.
- **$HOME** (`/global/homes/s/siddart2/`): Persistent but small. Scripts and configs.
- Inside container, `HOME` is overridden to `$SCRATCH`.
- poke-env is installed into site-packages via `pip install -e pokechamp`. No symlinks needed.

## SLURM Quick Reference

| QOS | Max Wall | Use |
|-----|----------|-----|
| `interactive` | 4h | `salloc` only (NOT sbatch) |
| `debug` | 30m | Quick tests |
| `regular` | 48h | Production training |

Account: `-A m5017` (or `-A m5017_g` with reservation).
Constraint: `-C "gpu&hbm80g"` for A100-80GB nodes.
