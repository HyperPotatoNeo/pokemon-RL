#!/bin/bash
# Generic RL training launch script for pokemon-rl + prime-rl.
#
# Usage:
#   bash scripts/launch_rl.sh [config_file]
#
# Default config: configs/pokemon/rl_selfplay.toml
#
# This script:
#   1. Starts Pokemon Showdown in the background
#   2. Activates the prime-rl virtualenv
#   3. Installs pokemon-rl + pokechamp if not already present
#   4. Runs the RL training pipeline via `rl @ <config>`
#
# Environment variables (override as needed):
#   POKEMON_RL_DIR  — path to pokemon-rl repo (auto-detected from script location)
#   PRIME_RL_DIR    — path to prime-rl repo (default: sibling directory ../prime-rl)
#   NODE_BIN        — path to Node.js binary (default: node)
#   SHOWDOWN_PORT   — Showdown server port (default: 8000)
#   INFERENCE_GPUS  — GPU IDs for vLLM inference (default: auto)
#   TRAINER_GPUS    — GPU IDs for trainer (default: auto)
set -e

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(dirname "$SCRIPT_DIR")}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$(dirname "$POKEMON_RL_DIR")/prime-rl}"
CONFIG="${1:-$POKEMON_RL_DIR/configs/pokemon/rl_selfplay.toml}"
NODE_BIN="${NODE_BIN:-node}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
SHOWDOWN_PATH="${SHOWDOWN_PATH:-$POKEMON_RL_DIR/vendor/pokechamp/pokemon-showdown/pokemon-showdown}"

echo "=== Pokemon RL Training ==="
echo "Config:      $CONFIG"
echo "Pokemon-RL:  $POKEMON_RL_DIR"
echo "Prime-RL:    $PRIME_RL_DIR"
echo "Showdown:    port $SHOWDOWN_PORT"

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

# --- 1. Start Showdown ---
echo "Starting Showdown on port $SHOWDOWN_PORT..."
"$NODE_BIN" "$SHOWDOWN_PATH" start --no-security --port "$SHOWDOWN_PORT" &
SHOWDOWN_PID=$!
trap "kill $SHOWDOWN_PID 2>/dev/null" EXIT

# Wait for Showdown to be ready
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

# --- 4. Symlink poke_env data (pokechamp uses relative paths) ---
if [ ! -e "$PRIME_RL_DIR/poke_env" ]; then
    ln -sfn "$POKEMON_RL_DIR/vendor/pokechamp/poke_env" "$PRIME_RL_DIR/poke_env"
    echo "Created poke_env symlink in prime-rl"
fi

# --- 5. Detect and start external opponents (e.g., kakuna) ---
OPPONENT_PID=""
OPPONENT_TYPE=$(python -c "
import tomllib, sys
with open('$CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
env_args = {}
for env in cfg.get('orchestrator', {}).get('env', []):
    env_args = env.get('args', {})
print(env_args.get('opponent_type', ''))
" 2>/dev/null)

if [ "$OPPONENT_TYPE" = "kakuna" ]; then
    KAKUNA_SCRIPT="${KAKUNA_LAUNCHER:-$POKEMON_RL_DIR/local_scripts/launch_kakuna_opponent.sh}"
    if [ -f "$KAKUNA_SCRIPT" ]; then
        echo "Starting Kakuna opponent process..."
        bash "$KAKUNA_SCRIPT" &
        OPPONENT_PID=$!
        trap "kill $OPPONENT_PID 2>/dev/null; kill $SHOWDOWN_PID 2>/dev/null" EXIT
        echo "Waiting 30s for Kakuna to initialize..."
        sleep 30
        if ! kill -0 "$OPPONENT_PID" 2>/dev/null; then
            echo "ERROR: Kakuna process died during startup" >&2
            exit 1
        fi
        echo "Kakuna running (PID: $OPPONENT_PID)"
    else
        echo "WARNING: Kakuna launcher not found at $KAKUNA_SCRIPT"
        echo "Start the Kakuna process manually before training begins."
    fi
fi

# --- 6. Run training ---
echo "Starting RL training..."
GPU_ARGS=""
[ -n "$INFERENCE_GPUS" ] && GPU_ARGS="$GPU_ARGS --inference_gpu_ids $INFERENCE_GPUS"
[ -n "$TRAINER_GPUS" ] && GPU_ARGS="$GPU_ARGS --trainer_gpu_ids $TRAINER_GPUS"

rl @ "$CONFIG" $GPU_ARGS
echo "=== Training complete ==="
