#!/bin/bash
# Generic eval launch script for pokemon-rl.
#
# Usage:
#   bash scripts/launch_eval.sh [config_file]
#
# Default config: configs/pokemon/eval_example.toml
#
# This script:
#   1. Starts Pokemon Showdown in the background
#   2. Activates the prime-rl virtualenv
#   3. Installs pokemon-rl + pokechamp if not already present
#   4. Starts agent vLLM server (with DP)
#   5. Runs the eval runner
#
# Environment variables (override as needed):
#   POKEMON_RL_DIR  — path to pokemon-rl repo (auto-detected)
#   PRIME_RL_DIR    — path to prime-rl repo (default: sibling ../prime-rl)
#   NODE_BIN        — path to Node.js binary (default: node)
#   SHOWDOWN_PORT   — Showdown server port (default: 8000)
#   AGENT_GPUS      — GPU IDs for agent vLLM (default: 0,1)
#   AGENT_PORT      — Agent vLLM port (default: 8001)
#   NODE_RANK       — Node rank for multi-node (default: 0)
#   N_NODES         — Total nodes (default: 1)
set -e

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(dirname "$SCRIPT_DIR")}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$(dirname "$POKEMON_RL_DIR")/prime-rl}"
CONFIG="${1:-$POKEMON_RL_DIR/configs/pokemon/eval_example.toml}"
NODE_BIN="${NODE_BIN:-node}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
SHOWDOWN_PATH="${SHOWDOWN_PATH:-$POKEMON_RL_DIR/vendor/pokechamp/pokemon-showdown/pokemon-showdown}"
AGENT_GPUS="${AGENT_GPUS:-0,1}"
AGENT_PORT="${AGENT_PORT:-8001}"
NODE_RANK="${NODE_RANK:-0}"
N_NODES="${N_NODES:-1}"

echo "=== Pokemon RL Eval ==="
echo "Config:      $CONFIG"
echo "Pokemon-RL:  $POKEMON_RL_DIR"
echo "Prime-RL:    $PRIME_RL_DIR"
echo "Showdown:    port $SHOWDOWN_PORT"
echo "Agent GPUs:  $AGENT_GPUS (port $AGENT_PORT)"
echo "Node:        $NODE_RANK / $N_NODES"

# --- Validate ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi
if [ ! -d "$PRIME_RL_DIR" ]; then
    echo "ERROR: prime-rl directory not found: $PRIME_RL_DIR" >&2
    exit 1
fi

# --- Cleanup trap ---
PIDS_TO_KILL=""
cleanup() {
    echo "Cleaning up..."
    for pid in $PIDS_TO_KILL; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- 1. Start Showdown ---
echo "Starting Showdown on port $SHOWDOWN_PORT..."
"$NODE_BIN" "$SHOWDOWN_PATH" start --no-security --port "$SHOWDOWN_PORT" &
SHOWDOWN_PID=$!
PIDS_TO_KILL="$SHOWDOWN_PID"

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

# --- 5. Parse agent model from config ---
AGENT_MODEL=$(python -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg.get('agent_model', ''))
" 2>/dev/null)

if [ -z "$AGENT_MODEL" ]; then
    echo "ERROR: agent_model not found in config" >&2
    exit 1
fi
echo "Agent model: $AGENT_MODEL"

# --- 6. Start agent vLLM ---
echo "Starting agent vLLM on GPUs $AGENT_GPUS, port $AGENT_PORT..."
NUM_AGENT_GPUS=$(echo "$AGENT_GPUS" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$AGENT_GPUS python -m vllm.entrypoints.openai.api_server \
    --model "$AGENT_MODEL" \
    --port "$AGENT_PORT" \
    --trust-remote-code \
    --data-parallel-size "$NUM_AGENT_GPUS" &
VLLM_PID=$!
PIDS_TO_KILL="$PIDS_TO_KILL $VLLM_PID"

echo "Waiting for agent vLLM to be ready..."
VLLM_READY=0
for i in $(seq 1 120); do
    if nc -z localhost "$AGENT_PORT" 2>/dev/null; then
        echo "Agent vLLM ready"
        VLLM_READY=1
        break
    fi
    sleep 2
done
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: Agent vLLM failed to start on port $AGENT_PORT within 240s" >&2
    exit 1
fi

# --- 7. Detect and start metamon opponents ---
HAS_METAMON=$(python -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
metamon = [o for o in cfg.get('opponents', []) if o.get('type') == 'metamon']
print(len(metamon))
" 2>/dev/null)

if [ "$HAS_METAMON" -gt 0 ]; then
    KAKUNA_SCRIPT="${KAKUNA_LAUNCHER:-$POKEMON_RL_DIR/local_scripts/launch_kakuna_opponent.sh}"
    if [ -f "$KAKUNA_SCRIPT" ]; then
        echo "Starting Kakuna opponent instances..."
        bash "$KAKUNA_SCRIPT" 8 &
        KAKUNA_PID=$!
        PIDS_TO_KILL="$PIDS_TO_KILL $KAKUNA_PID"
        echo "Waiting 45s for Kakuna initialization..."
        sleep 45
    else
        echo "WARNING: Kakuna launcher not found. Start metamon opponents manually."
    fi
fi

# --- 8. Run eval ---
echo "Starting eval..."
python -m pokemon_rl.eval.runner "$CONFIG" --node_rank "$NODE_RANK" --n_nodes "$N_NODES"
echo "=== Eval complete ==="
