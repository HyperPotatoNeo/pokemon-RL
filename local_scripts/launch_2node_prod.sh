#!/bin/bash
#SBATCH -A m5017
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH -J pokemon-2node
#SBATCH -o /pscratch/sd/s/siddart2/pokemon-rl/logs/2node_%j.out
#SBATCH -e /pscratch/sd/s/siddart2/pokemon-rl/logs/2node_%j.err
#
# Generic 2-node RL training launcher.
# Node 1: inference (DP=4, 4 GPUs)
# Node 2: trainer + Showdown + orchestrator (GPU split configurable)
#
# Usage: sbatch local_scripts/launch_2node_prod.sh <config_path> [inference_gpus] [trainer_gpus]
#   The config must use __INFERENCE_NODE__ placeholder in orchestrator.client.base_url.
#
#   Default GPU split: no inference on Node 2, trainer uses all 4 GPUs (4x4 split).
#   Override: pass inference_gpu_ids and trainer_gpu_ids as 2nd/3rd args.
#
# Examples:
#   sbatch local_scripts/launch_2node_prod.sh configs/pokemon/rl_vs_abyssal_600_4x4.toml
#   sbatch local_scripts/launch_2node_prod.sh configs/pokemon/rl_vs_abyssal_600.toml 0,1 2,3
#
# To add reservation: sbatch --reservation=_CAP_tinker local_scripts/launch_2node_prod.sh ...
set -e

# --- Parse args ---
CONFIG_PATH="${1:?Usage: sbatch launch_2node_prod.sh <config_path> [inference_gpus] [trainer_gpus]}"
INFERENCE_GPU_IDS="${2:-}"    # Empty = no inference on Node 2
TRAINER_GPU_IDS="${3:-0,1,2,3}"  # Default: all 4 GPUs for trainer

SCRATCH=/pscratch/sd/s/siddart2
POKEMON_RL=$SCRATCH/pokemon-rl
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

# Resolve config path (relative to pokemon-rl root or absolute)
if [[ "$CONFIG_PATH" = /* ]]; then
    CONFIG_ABS="$CONFIG_PATH"
else
    CONFIG_ABS="$POKEMON_RL/$CONFIG_PATH"
fi

if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: Config not found: $CONFIG_ABS"
    exit 1
fi

# Verify config has __INFERENCE_NODE__ placeholder
if ! grep -q "__INFERENCE_NODE__" "$CONFIG_ABS"; then
    echo "WARNING: Config does not contain __INFERENCE_NODE__ placeholder."
    echo "         Multi-node inference routing may not work correctly."
fi

# Parse SLURM nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE1=${NODES[0]}   # Inference only
NODE2=${NODES[1]}   # Trainer + Showdown + local inference

echo "============================================"
echo "Pokemon RL: 2-Node Training"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Config:     $CONFIG_ABS"
echo "Node 1 (inference):  $NODE1"
echo "Node 2 (rl+trainer): $NODE2"
echo "Inference GPUs (Node 2): ${INFERENCE_GPU_IDS:-none}"
echo "Trainer GPUs (Node 2):   $TRAINER_GPU_IDS"
echo "Start time: $(date)"
echo "============================================"

# Create logs directory
mkdir -p $POKEMON_RL/logs

# --- 1. Set up containers on both nodes (host networking for cross-node) ---
echo ""
echo "=== Setting up containers ==="

for NODE in $NODE1 $NODE2; do
    echo "Setting up $NODE..."
    ssh $NODE "export HOME=$SCRATCH && \
        export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
        bash $POKEMON_RL/local_scripts/setup_node_hostnet.sh false" &
done
wait
echo "Containers ready on both nodes"

# --- 2. Start inference on Node 1 (background) ---
echo ""
echo "=== Starting inference on $NODE1 ==="
ssh $NODE1 "export HOME=$SCRATCH && \
    export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
    podman-hpc exec skyrl bash $POKEMON_RL/local_scripts/abyssal_node1_inference.sh" \
    > $POKEMON_RL/logs/node1_inference_$SLURM_JOB_ID.log 2>&1 &
INFER_PID=$!

# Wait for inference server to be ready.
# The batch script runs on NODE1 (head), so check localhost directly (no SSH).
echo "Waiting for inference server on $NODE1:8001..."
INFER_READY=0
for i in $(seq 1 120); do
    if nc -z localhost 8001 2>/dev/null; then
        INFER_READY=1
        break
    fi
    # Also check if process died
    if ! kill -0 $INFER_PID 2>/dev/null; then
        echo "ERROR: Inference process died during startup. Check logs/node1_inference_$SLURM_JOB_ID.log"
        exit 1
    fi
    sleep 5
done

if [ "$INFER_READY" -eq 0 ]; then
    echo "WARNING: Inference not responding on port 8001 after 10 min, proceeding anyway..."
fi
echo "Inference server ready on $NODE1"

# --- 3. Generate resolved config and start RL on Node 2 ---
echo ""
echo "=== Starting RL training on $NODE2 ==="

# Write a small inner script that resolves __INFERENCE_NODE__ and runs RL.
# This avoids nested quoting issues with SSH + podman-hpc exec.
INNER_SCRIPT=/tmp/2node_rl_inner_${SLURM_JOB_ID}.sh
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
set -e
SCRATCH=/pscratch/sd/s/siddart2
POKEMON_RL=$SCRATCH/pokemon-rl
PRIME_RL=$SCRATCH/prime-rl
NODE_BIN=$SCRATCH/node-v20.18.1-linux-x64/bin/node
SHOWDOWN_PATH=$POKEMON_RL/vendor/pokechamp/pokemon-showdown/pokemon-showdown

# --- Start Showdown ---
echo "Starting Showdown server..."
export PATH=$SCRATCH/node-v20.18.1-linux-x64/bin:$PATH
cd $(dirname $SHOWDOWN_PATH)
nohup $NODE_BIN pokemon-showdown start --no-security > /tmp/showdown.log 2>&1 &
SHOWDOWN_PID=$!

for i in $(seq 1 30); do
    if curl -s http://localhost:8000 >/dev/null 2>&1; then
        echo "Showdown ready on port 8000"
        break
    fi
    sleep 1
done

# --- Activate prime-rl ---
cd $PRIME_RL
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

# Install pokemon-rl if needed
python -c "import pokemon_rl" 2>/dev/null || {
    echo "Installing pokemon-rl into prime-rl venv..."
    UV_CACHE_DIR=$SCRATCH/uv-cache uv pip install -e "$POKEMON_RL/vendor/pokechamp" 2>&1 | tail -3
    UV_CACHE_DIR=$SCRATCH/uv-cache uv pip install -e "$POKEMON_RL" 2>&1 | tail -3
}

# Symlink poke_env data
[ -e "$PRIME_RL/poke_env" ] || ln -sfn "$POKEMON_RL/vendor/pokechamp/poke_env" "$PRIME_RL/poke_env"

python -c "from pokemon_rl import load_environment; print('pokemon_rl OK')"

# --- Resolve config ---
CONFIG_RESOLVED=/tmp/resolved_config_${SLURM_JOB_ID}.toml
sed "s/__INFERENCE_NODE__/${INFERENCE_NODE}/g" "$CONFIG_SRC" > "$CONFIG_RESOLVED"

echo "=== Starting RL training ==="
echo "Host: $(hostname)"
echo "Inference Node: $INFERENCE_NODE"
echo "Config: $CONFIG_RESOLVED"

# --- Run training ---
RL_CMD="uv run rl @ $CONFIG_RESOLVED --trainer_gpu_ids $TRAINER_GPU_IDS"
if [ -n "$INFERENCE_GPU_IDS" ]; then
    RL_CMD="$RL_CMD --inference_gpu_ids $INFERENCE_GPU_IDS"
fi
echo "Running: $RL_CMD"
eval $RL_CMD

echo "=== Training complete ==="
kill $SHOWDOWN_PID 2>/dev/null || true
INNEREOF

ssh $NODE2 "export HOME=$SCRATCH && \
    export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
    podman-hpc exec -e INFERENCE_NODE=$NODE1 -e CONFIG_SRC=$CONFIG_ABS -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e INFERENCE_GPU_IDS=$INFERENCE_GPU_IDS -e TRAINER_GPU_IDS=$TRAINER_GPU_IDS skyrl \
    bash $INNER_SCRIPT" \
    > $POKEMON_RL/logs/node2_rl_$SLURM_JOB_ID.log 2>&1 &
RL_PID=$!

echo "Training launched (PID: $RL_PID)"
echo "Logs:"
echo "  Inference: $POKEMON_RL/logs/node1_inference_$SLURM_JOB_ID.log"
echo "  Training:  $POKEMON_RL/logs/node2_rl_$SLURM_JOB_ID.log"

# --- 4. Wait for training to finish ---
wait $RL_PID
RL_EXIT=$?

echo ""
echo "============================================"
echo "Training finished with exit code: $RL_EXIT"
echo "End time: $(date)"
echo "============================================"

# Clean up inference
kill $INFER_PID 2>/dev/null || true
rm -f "$INNER_SCRIPT"
exit $RL_EXIT
