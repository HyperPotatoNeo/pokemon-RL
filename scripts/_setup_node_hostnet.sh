#!/bin/bash
# Setup node with host networking (required for multi-node play)
set -e
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
# Use the pokechamp submodule (vendor/pokechamp) — has ws:// fix for cross-node
SHOWDOWN_PATH=$HOME/pokemon-rl/vendor/pokechamp/pokemon-showdown

cd $HOME

# Stop existing container
podman-hpc stop skyrl 2>/dev/null || true
sleep 2

# Start with host networking
echo "Starting skyrl container with --net=host..."
podman-hpc run --rm -d \
    --user "$(id -u):$(id -g)" --replace --name skyrl \
    --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
    --net=host \
    -e SCRATCH -e HOME \
    -v "$HOME":"$HOME" -w "$HOME/pokemon-rl" \
    docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
    sleep infinity 2>&1

SHOWDOWN=${1:-false}
if [ "$SHOWDOWN" = "true" ]; then
    echo "Starting Showdown..."
    podman-hpc exec skyrl bash -c "
        export PATH=/pscratch/sd/s/siddart2/node-v20.18.1-linux-x64/bin:\$PATH
        cd $SHOWDOWN_PATH
        if curl -s http://localhost:8000 >/dev/null 2>&1; then
            echo 'Showdown already running'
        else
            nohup node pokemon-showdown start --no-security > /tmp/showdown.log 2>&1 &
            for i in \$(seq 1 30); do
                if curl -s http://localhost:8000 >/dev/null 2>&1; then
                    echo 'Showdown ready on port 8000'
                    break
                fi
                sleep 1
            done
        fi
    "
fi

echo "Node ready: $(hostname)"
