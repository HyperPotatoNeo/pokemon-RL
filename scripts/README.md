# Scripts

Setup and test scripts for pokemon-rl. All scripts use environment variables with
sensible defaults — override them for your cluster environment.

## Files

| Script | Purpose |
|--------|---------|
| `setup_node.sh` | Install deps + start Showdown server |
| `setup_node_hostnet.sh` | Like setup_node but with host networking (for multi-node) |
| `run_tests.sh` | Run pytest inside the project venv |
| `launch_rl.sh` | Start Showdown + install deps + run RL training |
| `launch_interleaved.sh` | Start Showdown + vLLM + run interleaved trajectory RL training |
| `launch_eval.sh` | Start Showdown + agent vLLM + run eval runner |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POKEMON_RL_DIR` | Auto-detected from script location | Path to pokemon-rl repo |
| `NODE_BIN` | `node` | Path to Node.js binary |
| `CONTAINER_IMAGE` | `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8` | Container image |
| `CONTAINER_NAME` | `skyrl` | Container name |
| `SHOWDOWN_PORT` | `8000` | Showdown server port |
| `PRIME_RL_DIR` | `../prime-rl` | Path to prime-rl repo (for eval) |
| `AGENT_GPUS` | `0,1` | GPU IDs for agent vLLM (eval) |
| `AGENT_PORT` | `8001` | Agent vLLM port (eval) |
| `NODE_RANK` | `0` | Node rank for multi-node eval |
| `N_NODES` | `1` | Total nodes for multi-node eval |

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

## Launch Training

```bash
# Self-play training (default config):
bash scripts/launch_rl.sh

# Specific config:
bash scripts/launch_rl.sh configs/pokemon/rl_interleaved.toml

# Custom paths:
PRIME_RL_DIR=/path/to/prime-rl NODE_BIN=/usr/bin/node \
  bash scripts/launch_rl.sh configs/pokemon/rl_test.toml
```

See [docs/rl_training.md](../docs/rl_training.md) for full configuration reference.

## Launch Eval

```bash
# Default config:
bash scripts/launch_eval.sh

# Specific config:
bash scripts/launch_eval.sh configs/pokemon/eval_example.toml

# Custom GPU split + multi-node:
AGENT_GPUS=0,1,2,3 NODE_RANK=0 N_NODES=2 \
  bash scripts/launch_eval.sh configs/pokemon/eval_example.toml
```

See [docs/eval_testing_protocol.md](../docs/eval_testing_protocol.md) for the full testing guide.

## Cluster-Specific Scripts

For NERSC Perlmutter or other HPC environments, create cluster-specific scripts in
`local_scripts/` (gitignored). See `local_scripts/README.md` for the Perlmutter setup.
