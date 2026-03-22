#!/bin/bash
# Run tests inside the project venv.
# Usage: bash scripts/run_tests.sh [pytest args...]
# Examples:
#   bash scripts/run_tests.sh                    # All tests
#   bash scripts/run_tests.sh -m unit            # Unit tests only
#   bash scripts/run_tests.sh -m integration -v  # Integration tests verbose
set -e

POKEMON_RL_DIR="${POKEMON_RL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$POKEMON_RL_DIR"
source .venv/bin/activate
export SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
python -m pytest "${@:--v}"
