#!/bin/bash
# Set up the environment on a compute node.
# Starts a container, Showdown server, and installs pokemon-rl.
#
# Required env vars (or set defaults below):
#   POKEMON_RL_DIR  — path to pokemon-rl repo
#   NODE_BIN        — path to Node.js binary
#   CONTAINER_IMAGE — container image to use
#
# Usage: bash scripts/setup_node.sh
set -e

POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
POKECHAMP_PATH="$POKEMON_RL_DIR/vendor/pokechamp"
SHOWDOWN_PATH="$POKECHAMP_PATH/pokemon-showdown"
NODE_BIN="${NODE_BIN:-node}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8}"
CONTAINER_NAME="${CONTAINER_NAME:-skyrl}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"

# ---- 1. Clone pokemon-showdown if not present ----
if [ ! -d "$SHOWDOWN_PATH" ]; then
    echo "=== Cloning pokemon-showdown ==="
    git clone https://github.com/smogon/pokemon-showdown.git "$SHOWDOWN_PATH"
fi

# ---- 2. Start Showdown server ----
echo "=== Starting Showdown server ==="
NODE_DIR="$(dirname "$NODE_BIN")"
export PATH="$NODE_DIR:$PATH"
cd "$SHOWDOWN_PATH"
if curl -s "http://localhost:$SHOWDOWN_PORT" >/dev/null 2>&1; then
    echo "Showdown already running on port $SHOWDOWN_PORT"
else
    nohup "$NODE_BIN" pokemon-showdown start --no-security --port "$SHOWDOWN_PORT" > /tmp/showdown.log 2>&1 &
    echo "Waiting for Showdown to start..."
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$SHOWDOWN_PORT" >/dev/null 2>&1; then
            echo "Showdown ready on port $SHOWDOWN_PORT"
            break
        fi
        sleep 1
    done
fi

# ---- 3. Create venv and install deps ----
echo ""
echo "=== Installing pokemon-rl ==="
cd "$POKEMON_RL_DIR"

# Create isolated venv
python3 -m venv .venv 2>/dev/null || uv venv --python 3.12 .venv 2>/dev/null || true
source .venv/bin/activate

# Install pokechamp from submodule (brings poke_env fork + transitive deps)
echo "Installing pokechamp (brings poke-env, torch, etc.)..."
pip install -e "$POKECHAMP_PATH" 2>&1 | tail -5

# Install pokemon-rl itself
echo "Installing pokemon-rl..."
pip install -e ".[test]" 2>&1 | tail -5

# Symlink for pokechamp data_cache relative paths
ln -sf vendor/pokechamp/poke_env poke_env 2>/dev/null || true

echo ""
echo "Verifying imports..."
python -c 'from poke_env.player.player import Player; print("  poke-env: OK")'
python -c 'import poke_env; from pokechamp.prompts import state_translate; print("  pokechamp: OK")'
python -c 'from pokemon_rl.engine import ShowdownEngine; print("  pokemon_rl: OK")'

echo ""
echo "=== Setup complete ==="
echo "Run tests: .venv/bin/python -m pytest -v"
