# pokemon-rl

Pokemon Showdown multi-agent RL environment for [prime-rl](https://github.com/HyperPotatoNeo/prime-rl). Enables training LLMs to play competitive Pokemon via GRPO, with turn-by-turn control, self-play, and configurable rewards.

## Architecture

```
Layer 4: PokemonBattleEnv   — LLM harness (verifiers hooks, prompts, rewards)
         PokemonRubric      — Passthrough reward + game metrics
         _AgentContext       — Per-agent state during rollout
Layer 3: StateTranslator    — Battle state <-> LLM text (pokechamp format)
Layer 2: BattleManager      — Turn-by-turn battle orchestration
         BattleAdapter      — Full-battle mode (callback-driven)
         ControllablePlayer — Queue-based external control
Layer 1: ShowdownEngine     — Node.js Pokemon Showdown process manager
```

**BattleManager** is the general battle harness. It knows Pokemon, players, turns, and game state. It does NOT know about LLMs, prompts, tokens, or rewards.

**PokemonBattleEnv** is the LLM harness on top. Inherits from `vf.MultiTurnEnv` (when verifiers is installed) and implements 7 hook methods: `setup_state`, `get_prompt_messages`, `add_trajectory_step`, `render_completion`, `game_over` (@vf.stop), `cleanup_battle` (@vf.cleanup), `env_response`. Translates battle states to text prompts, parses text to actions, and assigns rewards with per-step advantage pre-setting for self-play.

See [docs/architecture.md](docs/architecture.md) for detailed data flow and design decisions.

## Installation

### Prerequisites

- Python 3.10+
- Node.js (for Pokemon Showdown server)
- [pokechamp](https://github.com/HyperPotatoNeo/pokechamp) — bundled as a git submodule at `vendor/pokechamp` (brings poke-env fork as a dependency)

### Install

```bash
git clone --recurse-submodules https://github.com/HyperPotatoNeo/pokemon-RL.git
cd pokemon-RL

# Create venv and install
python -m venv .venv
source .venv/bin/activate

# Install pokechamp submodule (brings poke-env fork + transitive deps)
pip install -e vendor/pokechamp

# Symlink for pokechamp_io data cache (hardcoded relative paths)
ln -sf vendor/pokechamp/poke_env poke_env

# Install pokemon-rl with test deps
pip install -e ".[test]"

# Optional: install verifiers for full RL pipeline integration
pip install "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@209774a"
```

Or with `uv` (faster):
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e vendor/pokechamp
ln -sf vendor/pokechamp/poke_env poke_env
uv pip install -e ".[test]"
```

### Start Showdown Server

```bash
cd /path/to/pokemon-showdown  # from pokechamp repo or smogon/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

### Run Tests

```bash
# Unit tests (no Showdown needed):
.venv/bin/python -m pytest -m unit -v

# All tests (Showdown must be running on port 8000):
.venv/bin/python -m pytest -v
```

For NERSC Perlmutter deployment, see [docs/deployment.md](docs/deployment.md).

## Usage

```python
# Single agent vs heuristic opponent
from pokemon_rl.env import PokemonBattleEnv
from pokemon_rl.adapter import random_action

env = PokemonBattleEnv(
    battle_format="gen1randombattle",
    port=8000,
    play_mode="single",
    observation_format="simple",
)
result = await env.run_turn_by_turn(action_fn=random_action)
# result: {"won": True, "turns": 23, "reward": 1.0, "trajectory": [...]}
```

```python
# Self-play (both agents train)
env = PokemonBattleEnv(
    battle_format="gen1randombattle",
    play_mode="self_play",
    observation_format="simple",
)
result = await env.run_turn_by_turn(action_fn=random_action)
# P0 wins: P0 steps get reward_win, P1 steps get reward_loss
# Advantages pre-set with config-derived baseline
```

```python
# Verifiers integration (prime-rl orchestrator)
# In TOML config:
# [[orchestrator.env]]
# id = "pokemon_rl"
# [orchestrator.env.args]
# battle_format = "gen1randombattle"
# play_mode = "self_play"
# [orchestrator]
# trajectory_strategy = "branching"
# rollouts_per_example = 4
```

See [docs/rewards.md](docs/rewards.md) for the configurable reward system.

## RL Training

Train LLMs to play Pokemon via GRPO with prime-rl integration.

### Quick Start

```bash
# Generic launch (inside prime-rl environment):
bash scripts/launch_rl.sh configs/pokemon/rl_test.toml

# Or manually (from prime-rl directory):
cd /path/to/prime-rl && source .venv/bin/activate
pip install -e /path/to/pokemon-rl/vendor/pokechamp
pip install -e /path/to/pokemon-rl
ln -sfn /path/to/pokemon-rl/vendor/pokechamp/poke_env poke_env  # required: pokechamp data cache
rl @ /path/to/pokemon-rl/configs/pokemon/rl_test.toml
```

### Configs

| Config | Mode | Strategy | Description |
|--------|------|----------|-------------|
| `rl_interleaved.toml` | single | interleaved | Production vs abyssal, full-conversation training (gen9ou, batch=64, seq_len=32K) |
| `rl_vs_abyssal_600_4x4.toml` | single | branching | Production vs abyssal, per-turn training (gen9ou, multi-node) |
| `rl_test.toml` | self_play | branching | Integration testing (3 steps, batch=4) |
| `inference_interleaved.toml` | -- | -- | Standalone inference for interleaved (DP=4, 32K context, prefix caching) |

**Critical**: Every config must set `trajectory_strategy` explicitly:
- `"branching"` -- each turn becomes a separate TrainingSample (used for self-play and short-sequence training).
- `"interleaved"` -- the full conversation becomes one TrainingSample via `interleave_rollout`. Two LLM calls per turn (reasoning + extraction), lighter prompts after turn 1. Requires `interleaved=true` in env args and `seq_len=32768`. See [docs/rl_training.md](docs/rl_training.md) for the full comparison.

See [docs/rl_training.md](docs/rl_training.md) for the full RL training guide: config reference, prompt construction, launching, and architecture.

## Evaluation

Evaluate trained agents against heuristic, metamon, or LLM opponents.

```bash
# Generic launch (inside container/environment with venv active):
bash scripts/launch_eval.sh configs/pokemon/eval_example.toml

# NERSC production (sbatch with container):
sbatch local_scripts/launch_eval_prod.sh configs/pokemon/eval_example.toml

# Or run the eval runner directly (Showdown + agent vLLM must already be running):
python -m pokemon_rl.eval.runner configs/pokemon/eval_example.toml
```

### Eval Config

```toml
agent_model = "Qwen/Qwen3-4B-Instruct-2507"
agent_base_url = "http://localhost:8001/v1"
battle_format = "gen9ou"
n_battles_per_opp = 100
max_concurrent_battles = 8
observation_format = "pokechamp_io"
output_dir = "eval_outputs/my_eval"

[[opponents]]
name = "abyssal"
type = "heuristic"
heuristic = "abyssal"

[[opponents]]
name = "qwen2.5-1.5b"
type = "llm"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
base_url = "http://localhost:8002/v1"
gpu_ids = [2, 3]
```

Opponent types: `heuristic` (abyssal, max_damage, random), `metamon` (kakuna), `llm` (any vLLM-served model). The runner manages LLM opponent vLLM lifecycle automatically when `gpu_ids` is specified.

Output: per-opponent JSONL results + summary table with win/loss/draw rates, stderr, and avg game turns.

See [docs/eval_testing_protocol.md](docs/eval_testing_protocol.md) for the full eval testing protocol.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | 4-layer design, data flow diagrams, file map |
| [docs/concurrency.md](docs/concurrency.md) | POKE_LOOP bridging, cross-loop patterns, relay queue |
| [docs/rewards.md](docs/rewards.md) | Configurable rewards, advantage pre-setting, self-play assignment |
| [docs/selfplay.md](docs/selfplay.md) | Self-play mechanics, force-switch handling, hooks buffering |
| [docs/testing.md](docs/testing.md) | Test philosophy, markers, running tests, mock patterns |
| [docs/rl_training.md](docs/rl_training.md) | RL training setup, configs, prompt construction, launching |
| [docs/deployment.md](docs/deployment.md) | Cluster deployment, containers, multi-node, RL deployment |
| [docs/eval_testing_protocol.md](docs/eval_testing_protocol.md) | Eval testing protocol, commands, debugging playbook |
| [docs/api.md](docs/api.md) | Public API reference for all classes and methods |

## Project Status

**Phase 5 RL training + eval feature complete. Interleaved trajectory validated.** 400 tests passing (352 existing + 48 interleaved). Production training active (Qwen3-4B vs abyssal, 2 nodes). Interleaved mode: full-conversation training with two-phase LLM calls, entropy declining after 4 steps. Eval framework supports heuristic, metamon, and LLM-vs-LLM opponents with per-opponent statistics.

See [PROGRESS.md](PROGRESS.md) for changelog and [TODO.md](TODO.md) for roadmap.

## Dependencies

- **[poke-env](https://github.com/hsahovic/poke-env)** (installed via pokechamp) — Pokemon Showdown client library. Provides the `Player` base class, `Battle` objects, `BattleOrder` action representation, and WebSocket communication with Showdown. pokemon-rl's `ControllablePlayer` subclasses poke-env's `Player` to invert the callback-driven model into imperative queue-based control. All battle state (available moves, HP, types, etc.) comes from poke-env's `Battle` objects.

- **[pokechamp](https://github.com/HyperPotatoNeo/pokechamp)** (submodule at `vendor/pokechamp`) — LLM Pokemon battle agent with a poke-env fork. pokemon-rl uses pokechamp for two things: (1) the `"pokechamp_io"` prompt format in `StateTranslator`, which calls pokechamp's `state_translate` + `LocalSim` to produce rich prompts with damage calculations, and (2) as the installation vehicle for poke-env and its transitive dependencies (torch, etc.). Installing pokechamp via `pip install -e` puts poke-env into site-packages where it's importable normally. The fork includes a `ws://` fix for cross-node WebSocket connections to self-hosted Showdown servers.

- **[verifiers](https://github.com/PrimeIntellect-ai/verifiers)** (optional) — RL environment framework for prime-rl. When installed, `PokemonBattleEnv` inherits from `vf.MultiTurnEnv` and integrates with the orchestrator's scoring pipeline. `PokemonRubric` provides passthrough rewards and game metrics. Without verifiers, the env works standalone for testing.

- **[Pokemon Showdown](https://github.com/smogon/pokemon-showdown)** — Node.js battle server. `ShowdownEngine` manages the server process. Battles run locally via WebSocket (no internet connection needed). Requires Node.js.

- **pytest**, **pytest-asyncio** — Testing

No direct numpy, torch, or ML framework dependencies. The env produces text prompts and consumes text responses — the LLM/RL framework is plugged in externally via prime-rl's verifiers.
