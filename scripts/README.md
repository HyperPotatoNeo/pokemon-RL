# Scripts (Cluster-Specific)

This directory is gitignored. Each user creates their own scripts for
their cluster environment. Below are templates for NERSC Perlmutter.

## Files

| Script | Purpose |
|--------|---------|
| `allocate.sh` | Get a compute node from the reservation |
| `setup_node.sh` | Start container + Showdown server on compute node |
| `run_tests.sh` | Run tests inside the container |

## Usage

```bash
# 1. Get a compute node
source scripts/allocate.sh

# 2. On the compute node, set up the environment
bash scripts/setup_node.sh

# 3. Run tests
bash scripts/run_tests.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHOWDOWN_PATH` | `/pscratch/sd/s/siddart2/pokechamp/pokemon-showdown` | Path to Showdown |
| `NODE_PATH` | `/pscratch/sd/s/siddart2/node-v20.18.1-linux-x64/bin/node` | Node.js binary |
| `POKECHAMP_PATH` | `/pscratch/sd/s/siddart2/pokechamp` | Pokechamp repo |
| `METAMON_PATH` | `/pscratch/sd/s/siddart2/metamon` | Metamon repo |
| `SHOWDOWN_PORT` | `8000` | Showdown server port |
