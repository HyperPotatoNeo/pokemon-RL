#!/bin/bash
# Setup node with host networking (required for multi-node play).
# Like setup_node.sh but uses --net=host on the container so other nodes
# can reach this node's Showdown server via hostname:port.
#
# Required env vars (or set defaults below):
#   POKEMON_RL_DIR  — path to pokemon-rl repo
#   NODE_BIN        — path to Node.js binary
#   CONTAINER_IMAGE — container image to use
#
# Usage: bash scripts/setup_node_hostnet.sh [true]  # pass "true" to start Showdown
set -e

POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SHOWDOWN_PATH="$POKEMON_RL_DIR/vendor/pokechamp/pokemon-showdown"
NODE_BIN="${NODE_BIN:-node}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8}"
CONTAINER_NAME="${CONTAINER_NAME:-skyrl}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"

# Clone pokemon-showdown if not present
if [ ! -d "$SHOWDOWN_PATH" ]; then
    git clone https://github.com/smogon/pokemon-showdown.git "$SHOWDOWN_PATH"
fi

echo "Starting container with --net=host..."
# NOTE: Replace this section with your container runtime (docker, podman, etc.)
# The key flag is --net=host for cross-node connectivity.
echo "ERROR: Container launch command not configured."
echo "Edit this script for your environment (docker run / podman run / etc.)"
echo "Required flags: --net=host, GPU passthrough, volume mounts"
exit 1

# After container is running, optionally start Showdown:
SHOWDOWN=${1:-false}
if [ "$SHOWDOWN" = "true" ]; then
    echo "Starting Showdown..."
    NODE_DIR="$(dirname "$NODE_BIN")"
    export PATH="$NODE_DIR:$PATH"
    cd "$SHOWDOWN_PATH"
    if curl -s "http://localhost:$SHOWDOWN_PORT" >/dev/null 2>&1; then
        echo "Showdown already running"
    else
        nohup "$NODE_BIN" pokemon-showdown start --no-security --port "$SHOWDOWN_PORT" > /tmp/showdown.log 2>&1 &
        for i in $(seq 1 30); do
            if curl -s "http://localhost:$SHOWDOWN_PORT" >/dev/null 2>&1; then
                echo "Showdown ready on port $SHOWDOWN_PORT"
                break
            fi
            sleep 1
        done
    fi
fi

echo "Node ready: $(hostname)"
