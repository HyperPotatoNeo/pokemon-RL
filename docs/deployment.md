# Deployment

pokemon-rl can run on any system with Python 3.10+, Node.js, and optionally GPUs for LLM inference. This guide covers general setup and HPC cluster deployment.

## Quick Start (Any System)

```bash
git clone --recurse-submodules https://github.com/HyperPotatoNeo/pokemon-RL.git
cd pokemon-RL
bash scripts/setup_node.sh
bash scripts/run_tests.sh -m unit -v
```

## Environment Variables

All scripts use environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `POKEMON_RL_DIR` | Auto-detected | Path to pokemon-rl repo |
| `NODE_BIN` | `node` | Path to Node.js binary |
| `SHOWDOWN_PORT` | `8000` | Showdown server port |
| `SHOWDOWN_PATH` | `vendor/pokechamp/pokemon-showdown` | Showdown directory (env override for tests) |
| `NODE_PATH` | `node` | Node.js binary (env override for tests) |
| `VLLM_HOST` | `localhost` | vLLM server hostname (GPU tests) |
| `VLLM_PORT` | `8001` | vLLM server port (GPU tests) |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Model for GPU tests |
| `REMOTE_NODE` | unset | Remote hostname for multinode tests |

## Package Installation

```bash
cd pokemon-rl
source .venv/bin/activate

pip install -e vendor/pokechamp    # poke-env fork (ws:// fix) + deps
pip install -e ".[test]"           # pokemon-rl + pytest
```

**Always use the submodule** (`vendor/pokechamp`). It contains the poke-env fork with the `ws://` fix for non-localhost Showdown servers. The upstream poke-env uses `wss://` for non-localhost, which hangs on self-hosted servers without SSL.

## Showdown Server

Pokemon Showdown is a Node.js application. `setup_node.sh` clones it into `vendor/pokechamp/pokemon-showdown` (gitignored) if not already present.

```bash
# Manual start:
cd vendor/pokechamp/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

## Multi-Node Battles

For cross-node play (e.g., 2 GPU nodes running different models):

1. **Host networking required.** Default container bridge networking isolates the network — players can't reach Showdown on another node. Use `--net=host` on your container.

2. **One Showdown server, all players connect to it:**
   ```python
   BattleManager(server_host="other-hostname", port=8000)
   ```

3. **ws:// fix in pokechamp fork.** poke-env's `listen()` uses `wss://` (SSL) for any non-localhost server. Self-hosted Showdown doesn't have SSL, causing the WebSocket to hang on TLS handshake. The pokechamp fork at `vendor/pokechamp/poke_env/ps_client/ps_client.py:376` uses `ws://` instead, making cross-node battles work natively.

## GPU Tests (vLLM)

The GPU tests (`tests/test_phase4_gpu.py`) need vLLM serving a model:

```bash
# Start vLLM:
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 --max-model-len 4096 \
    --no-enable-log-requests &

# Run GPU tests:
VLLM_HOST=localhost VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
SHOWDOWN_PORT=8000 python -m pytest tests/test_phase4_gpu.py -v

# Multi-node tests (from a different node):
REMOTE_NODE=showdown-hostname \
  python -m pytest tests/test_phase4_gpu.py -k multinode -v
```

## HPC / Container Notes

For HPC clusters with container runtimes (podman, Singularity, etc.):

- Use `--net=host` for multi-node
- Use `--gpu` flags for GPU passthrough
- Mount your scratch filesystem into the container
- Create cluster-specific scripts in `local_scripts/` (gitignored)
- The `poke_env` symlink at project root is required for pokechamp's `data_cache.py` which uses hardcoded relative paths (`./poke_env/data/static/...`)

## Filesystem Notes

- poke-env is installed into site-packages via `pip install -e vendor/pokechamp`.
- A `poke_env` **symlink** exists at the project root → `vendor/pokechamp/poke_env`. This is required because pokechamp's `data_cache.py` uses relative paths (`./poke_env/data/static/...`). The symlink shadows the site-packages poke_env, which causes a circular import if you `from pokechamp.prompts import ...` directly. **Always `import poke_env` first** to break the cycle (the translator does this automatically).

## RL Training Deployment

See [rl_training.md](rl_training.md) for the full training guide with config reference.

### Port Allocation

Two servers run simultaneously:
- **Showdown server**: Port 8000 (default). Background Node.js process.
- **Inference server (vLLM)**: Different port (8001 recommended). prime-rl starts this automatically when `[inference]` is present in the TOML config.

Port 8000 is reserved for Showdown. Never configure the inference server on port 8000.

### Single-Node Layout

```
GPU 0-2: vLLM inference (tensor-parallel)
GPU 3:   Trainer (GRPO weight updates)
CPU:     Showdown + Orchestrator (background)
```

Use `scripts/launch_rl.sh` or `local_scripts/launch_1node.sh` (cluster-specific).

### Two-Node Layout

```
Node 0: Showdown :8000 + vLLM :8001 (4 GPUs)
Node 1: Orchestrator + Trainer (4 GPUs)
```

Requires `--net=host` container networking. See `local_scripts/launch_2node.sh`.

### Installing pokemon-rl into prime-rl

pokemon-rl must be installed into prime-rl's venv (not its own `.venv`):

```bash
cd /path/to/prime-rl && source .venv/bin/activate
pip install -e /path/to/pokemon-rl/vendor/pokechamp  # poke-env fork
pip install -e /path/to/pokemon-rl                    # pokemon-rl itself
ln -sfn /path/to/pokemon-rl/vendor/pokechamp/poke_env poke_env  # data symlink
```

The `scripts/launch_rl.sh` script does this automatically.

### Gradient Checkpointing

For single-GPU training (3 GPUs inference + 1 training on a 4-GPU node):

```toml
[trainer.model.ac]
freq = 1   # Full gradient checkpointing — prevents OOM on A100-80GB
```

Without this, Qwen3-4B training OOMs at ~76.5 GiB peak memory.
