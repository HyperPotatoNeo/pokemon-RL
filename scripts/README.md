# Scripts

Setup and test scripts for pokemon-rl. All scripts use environment variables with
sensible defaults — override them for your cluster environment.

## Files

| Script | Purpose |
|--------|---------|
| `setup_node.sh` | Install deps + start Showdown server |
| `setup_node_hostnet.sh` | Like setup_node but with host networking (for multi-node) |
| `run_tests.sh` | Run pytest inside the project venv |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POKEMON_RL_DIR` | Auto-detected from script location | Path to pokemon-rl repo |
| `NODE_BIN` | `node` | Path to Node.js binary |
| `CONTAINER_IMAGE` | `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8` | Container image |
| `CONTAINER_NAME` | `skyrl` | Container name |
| `SHOWDOWN_PORT` | `8000` | Showdown server port |

## Usage

```bash
# 1. Set up environment (installs from vendor/pokechamp submodule)
bash scripts/setup_node.sh

# 2. Run tests
bash scripts/run_tests.sh -v
bash scripts/run_tests.sh -m unit -v        # unit tests only
bash scripts/run_tests.sh -m integration -v # needs Showdown running

# 3. GPU tests (need vLLM serving a model)
VLLM_HOST=localhost VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  bash scripts/run_tests.sh tests/test_phase4_gpu.py -v

# 4. Multi-node tests (need --net=host container + REMOTE_NODE)
REMOTE_NODE=other-hostname \
  bash scripts/run_tests.sh tests/test_phase4_gpu.py -k multinode -v
```

## Cluster-Specific Scripts

For NERSC Perlmutter or other HPC environments, create cluster-specific scripts in
`local_scripts/` (gitignored). See `local_scripts/README.md` for the Perlmutter setup.
