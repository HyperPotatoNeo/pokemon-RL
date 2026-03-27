#!/bin/bash
# Generic launch script for interleaved trajectory RL training.
#
# Usage:
#   bash scripts/launch_interleaved.sh [config_file]
#
# Default config: configs/pokemon/rl_interleaved.toml
#
# This script:
#   1. Starts Pokemon Showdown in the background
#   2. Starts a vLLM inference server with prefix caching (interleaved needs long contexts)
#   3. Activates the prime-rl virtualenv
#   4. Installs pokemon-rl + pokechamp if not already present
#   5. Resolves config (replaces __INFER_1__ with localhost)
#   6. Runs the RL training pipeline via `rl @ <config>`
#
# Environment variables (override as needed):
#   POKEMON_RL_DIR  — path to pokemon-rl repo (auto-detected from script location)
#   PRIME_RL_DIR    — path to prime-rl repo (default: sibling directory ../prime-rl)
#   NODE_BIN        — path to Node.js binary (default: node)
#   SHOWDOWN_PORT   — Showdown server port (default: 8000)
#   VLLM_PORT       — vLLM inference server port (default: 8001)
#   INFERENCE_GPUS  — GPU IDs for vLLM inference (default: "0,1")
#   TRAINER_GPUS    — GPU IDs for trainer (default: "2,3")
#   MAX_MODEL_LEN   — vLLM max model length (default: 32768)
#   GPU_MEM_UTIL    — vLLM GPU memory utilization (default: 0.9)
#   MODEL_NAME      — HuggingFace model name (auto-detected from config)
set -e

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(dirname "$SCRIPT_DIR")}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$(dirname "$POKEMON_RL_DIR")/prime-rl}"
CONFIG="${1:-$POKEMON_RL_DIR/configs/pokemon/rl_interleaved.toml}"
NODE_BIN="${NODE_BIN:-node}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
SHOWDOWN_PATH="${SHOWDOWN_PATH:-$POKEMON_RL_DIR/vendor/pokechamp/pokemon-showdown/pokemon-showdown}"
VLLM_PORT="${VLLM_PORT:-8001}"
INFERENCE_GPUS="${INFERENCE_GPUS:-0,1}"
TRAINER_GPUS="${TRAINER_GPUS:-2,3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# Count inference GPUs for DP
NUM_INFER_GPUS=$(echo "$INFERENCE_GPUS" | tr ',' '\n' | wc -l)

echo "=== Pokemon RL: Interleaved Trajectory Training ==="
echo "Config:          $CONFIG"
echo "Pokemon-RL:      $POKEMON_RL_DIR"
echo "Prime-RL:        $PRIME_RL_DIR"
echo "Showdown:        port $SHOWDOWN_PORT"
echo "vLLM:            port $VLLM_PORT (GPUs: $INFERENCE_GPUS, DP=$NUM_INFER_GPUS)"
echo "Trainer GPUs:    $TRAINER_GPUS"
echo "Max model len:   $MAX_MODEL_LEN"

# --- Validate ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi
if [ ! -d "$PRIME_RL_DIR" ]; then
    echo "ERROR: prime-rl directory not found: $PRIME_RL_DIR" >&2
    echo "Set PRIME_RL_DIR to the correct path." >&2
    exit 1
fi
if ! command -v "$NODE_BIN" &>/dev/null && [ ! -x "$NODE_BIN" ]; then
    echo "ERROR: Node.js not found at: $NODE_BIN" >&2
    echo "Set NODE_BIN to the path of your node binary." >&2
    exit 1
fi

# --- Cleanup handler ---
PIDS_TO_KILL=()
cleanup() {
    echo "Cleaning up..."
    for PID in "${PIDS_TO_KILL[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- 1. Start Showdown ---
echo ""
echo "Starting Showdown on port $SHOWDOWN_PORT..."
"$NODE_BIN" "$SHOWDOWN_PATH" start --no-security --port "$SHOWDOWN_PORT" &
SHOWDOWN_PID=$!
PIDS_TO_KILL+=($SHOWDOWN_PID)

SHOWDOWN_READY=0
for i in $(seq 1 30); do
    if nc -z localhost "$SHOWDOWN_PORT" 2>/dev/null; then
        echo "Showdown ready"
        SHOWDOWN_READY=1
        break
    fi
    sleep 1
done
if [ "$SHOWDOWN_READY" -eq 0 ]; then
    echo "ERROR: Showdown failed to start on port $SHOWDOWN_PORT within 30s" >&2
    exit 1
fi

# --- 2. Activate prime-rl environment ---
cd "$PRIME_RL_DIR"
source .venv/bin/activate

# --- 3. Install pokemon-rl (if not already in the venv) ---
python -c "import pokemon_rl" 2>/dev/null || {
    echo "Installing pokemon-rl into prime-rl venv..."
    pip install -e "$POKEMON_RL_DIR/vendor/pokechamp" 2>&1 | tail -2
    pip install -e "$POKEMON_RL_DIR" 2>&1 | tail -2
}
python -c "from pokemon_rl import load_environment; print('pokemon_rl OK')"

# --- 4. Symlink poke_env data ---
if [ ! -e "$PRIME_RL_DIR/poke_env" ]; then
    ln -sfn "$POKEMON_RL_DIR/vendor/pokechamp/poke_env" "$PRIME_RL_DIR/poke_env"
    echo "Created poke_env symlink in prime-rl"
fi

# --- 5. Extract model name from config ---
MODEL_NAME="${MODEL_NAME:-$(python -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg.get('model', {}).get('name', 'Qwen/Qwen3-4B-Instruct-2507'))
")}"
echo "Model: $MODEL_NAME"

# --- 6. Start vLLM inference server ---
echo ""
echo "Starting vLLM inference server (DP=$NUM_INFER_GPUS, max_model_len=$MAX_MODEL_LEN)..."
CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --data-parallel-size "$NUM_INFER_GPUS" \
    --disable-log-requests \
    > /tmp/vllm_interleaved.log 2>&1 &
VLLM_PID=$!
PIDS_TO_KILL+=($VLLM_PID)

VLLM_READY=0
echo "Waiting for vLLM to start (up to 120s)..."
for i in $(seq 1 24); do
    sleep 5
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s"
        VLLM_READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Check /tmp/vllm_interleaved.log" >&2
        tail -20 /tmp/vllm_interleaved.log >&2
        exit 1
    fi
    echo "  ...waiting ($((i*5))s)"
done
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: vLLM failed to start within 120s. Check /tmp/vllm_interleaved.log" >&2
    tail -20 /tmp/vllm_interleaved.log >&2
    exit 1
fi

# --- 7. Resolve config (replace __INFER_1__ and __INFER_2__ with localhost) ---
CONFIG_RESOLVED="$(dirname "$CONFIG")/$(basename "$CONFIG" .toml)_resolved_$$.toml"
sed -e 's/__INFER_1__/localhost/g' -e 's/__INFER_2__/localhost/g' "$CONFIG" > "$CONFIG_RESOLVED"
echo "Resolved config: $CONFIG_RESOLVED"

# --- 8. Run training ---
echo ""
echo "Starting interleaved RL training..."
rl @ "$CONFIG_RESOLVED" \
    --trainer_gpu_ids "$TRAINER_GPUS"

echo "=== Interleaved training complete ==="
rm -f "$CONFIG_RESOLVED"
