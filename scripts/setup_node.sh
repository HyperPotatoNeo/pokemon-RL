#!/bin/bash
# Set up the environment on a compute node.
# Starts the skyrl container, Showdown server, and installs pokemon-rl.
# Usage: bash scripts/setup_node.sh
set -e

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
# Use the pokechamp submodule (vendor/pokechamp) — has ws:// fix for cross-node
POKECHAMP_PATH=$HOME/pokemon-rl/vendor/pokechamp
SHOWDOWN_PATH=$POKECHAMP_PATH/pokemon-showdown

cd $HOME

# ---- 1. Start the container ----
echo "=== Starting skyrl container ==="
podman-hpc run --rm -d \
    --user "$(id -u):$(id -g)" \
    --replace \
    --name skyrl \
    --group-add keep-groups \
    --userns keep-id \
    --gpu \
    --nccl \
    --shm-size=8g \
    -e SCRATCH -e HOME \
    -v "$HOME":"$HOME" \
    -w "$HOME/pokemon-rl" \
    docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
    sleep infinity 2>&1

echo "Container status:"
podman-hpc ps --filter name=skyrl 2>&1

# ---- 2. Start Showdown server inside container ----
echo ""
echo "=== Starting Showdown server ==="
podman-hpc exec skyrl bash -c "
    export PATH=/pscratch/sd/s/siddart2/node-v20.18.1-linux-x64/bin:\$PATH
    cd $SHOWDOWN_PATH
    if curl -s http://localhost:8000 >/dev/null 2>&1; then
        echo 'Showdown already running on port 8000'
    else
        nohup node pokemon-showdown start --no-security > /tmp/showdown.log 2>&1 &
        echo 'Waiting for Showdown to start...'
        for i in \$(seq 1 30); do
            if curl -s http://localhost:8000 >/dev/null 2>&1; then
                echo 'Showdown ready on port 8000'
                break
            fi
            sleep 1
        done
    fi
"

# ---- 3. Create venv and install deps ----
echo ""
echo "=== Installing pokemon-rl ==="
podman-hpc exec skyrl bash -c "
    cd $HOME/pokemon-rl
    export UV_CACHE_DIR=$HOME/uv-cache

    # Create isolated venv
    uv venv --python 3.12 .venv 2>/dev/null || true
    source .venv/bin/activate

    # Install pokechamp from local path (brings poke_env + all transitive deps)
    echo 'Installing pokechamp (brings poke-env, torch, etc.)...'
    uv pip install -e $POKECHAMP_PATH 2>&1 | tail -5

    # Install pokemon-rl itself
    echo 'Installing pokemon-rl...'
    uv pip install -e '.[test]' 2>&1 | tail -5

    echo ''
    echo 'Verifying imports...'
    python -c 'from poke_env.player.player import Player; print(\"  poke-env: OK\")'
    python -c 'import poke_env; from pokechamp.prompts import state_translate; print(\"  pokechamp: OK\")'
    python -c 'from pokemon_rl.engine import ShowdownEngine; print(\"  pokemon_rl: OK\")'
"

echo ""
echo "=== Setup complete ==="
echo "Run tests: bash scripts/run_tests.sh"
