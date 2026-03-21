# pokemon-rl

Pokemon Showdown multi-agent RL environment for [prime-rl](https://github.com/HyperPotatoNeo/prime-rl). Enables training LLMs to play competitive Pokemon via GRPO, with turn-by-turn control, self-play, and configurable rewards.

## Architecture

```
Layer 4: PokemonBattleEnv   — LLM harness (verifiers hooks, prompts, rewards)
Layer 3: StateTranslator    — Battle state <-> LLM text (pokechamp format)
Layer 2: BattleManager      — Turn-by-turn battle orchestration
         BattleAdapter      — Full-battle mode (callback-driven)
         ControllablePlayer — Queue-based external control
Layer 1: ShowdownEngine     — Node.js Pokemon Showdown process manager
```

**BattleManager** is the general battle harness. It knows Pokemon, players, turns, and game state. It does NOT know about LLMs, prompts, tokens, or rewards.

**PokemonBattleEnv** is the LLM harness on top. It translates battle states to text prompts, parses text to actions, and assigns rewards. Any LLM-in-the-loop use goes through here.

See [docs/architecture.md](docs/architecture.md) for detailed data flow and design decisions.

## Installation

### Prerequisites

- Python 3.10+
- Node.js (for Pokemon Showdown server)
- [pokechamp](https://github.com/HyperPotatoNeo/pokechamp) cloned locally (brings poke-env as a bundled dependency)

### Install

```bash
git clone https://github.com/HyperPotatoNeo/pokemon-RL.git
cd pokemon-RL

# Create venv and install
python -m venv .venv
source .venv/bin/activate

# Install pokechamp (brings poke-env + transitive deps like torch)
pip install -e /path/to/pokechamp

# Install pokemon-rl with test deps
pip install -e ".[test]"
```

Or with `uv` (faster):
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e /path/to/pokechamp
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
# Turn-by-turn with heuristic opponent
from pokemon_rl.env import PokemonBattleEnv
from pokemon_rl.translator import StateTranslator
from pokemon_rl.adapter import random_action

env = PokemonBattleEnv(
    translator=StateTranslator(format_style="simple"),
    control_mode="turn_by_turn",
    port=8000,
    battle_format="gen1randombattle",
    reward_win=1.0,
    reward_loss=0.0,
)
result = await env.run_turn_by_turn(action_fn=random_action)
# result: {"won": True, "turns": 23, "reward": 1.0, "trajectory": [...]}
```

```python
# Self-play
env = PokemonBattleEnv(
    translator=StateTranslator(format_style="simple"),
    control_mode="turn_by_turn",
    opponent_mode="self_play",
    reward_win=1.0,
    reward_loss=-1.0,  # symmetric rewards
)
result = await env.run_turn_by_turn(action_fn=random_action)
# P1 wins: P1 steps get 1.0, P2 steps get -1.0
```

See [docs/rewards.md](docs/rewards.md) for the configurable reward system.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | 4-layer design, data flow diagrams, file map |
| [docs/concurrency.md](docs/concurrency.md) | POKE_LOOP bridging, cross-loop patterns, relay queue |
| [docs/rewards.md](docs/rewards.md) | Configurable rewards, step rewards, self-play assignment |
| [docs/selfplay.md](docs/selfplay.md) | Self-play mechanics, force-switch handling, hooks buffering |
| [docs/testing.md](docs/testing.md) | Test philosophy, markers, running tests, mock patterns |
| [docs/deployment.md](docs/deployment.md) | NERSC Perlmutter setup, containers, multi-node battles |
| [docs/api.md](docs/api.md) | Public API reference for all classes and methods |

## Project Status

**149 tests** (122 unit + 27 integration), all passing.

See [PROGRESS.md](PROGRESS.md) for changelog and [TODO.md](TODO.md) for roadmap.

## Dependencies

- **[poke-env](https://github.com/hsahovic/poke-env)** (installed via pokechamp) — Pokemon Showdown client library. Provides the `Player` base class, `Battle` objects, `BattleOrder` action representation, and WebSocket communication with Showdown. pokemon-rl's `ControllablePlayer` subclasses poke-env's `Player` to invert the callback-driven model into imperative queue-based control. All battle state (available moves, HP, types, etc.) comes from poke-env's `Battle` objects.

- **[pokechamp](https://github.com/HyperPotatoNeo/pokechamp)** — LLM Pokemon battle agent. pokemon-rl uses pokechamp for two things: (1) the `"pokechamp_io"` prompt format in `StateTranslator`, which calls pokechamp's `state_translate` + `LocalSim` to produce rich prompts with damage calculations, and (2) as the installation vehicle for poke-env and its transitive dependencies (torch, etc.). Installing pokechamp via `pip install -e` puts poke-env into site-packages where it's importable normally.

- **[metamon](https://github.com/hsahovic/metamon)** — Pokemon battle baselines. The multi-node battle scripts (`scripts/_multinode_p1.py`, `_multinode_p2.py`) use metamon's `EmeraldKaizo` baseline for cross-node testing. metamon provides heuristic opponents stronger than poke-env's built-in `RandomPlayer` and `MaxBasePowerPlayer`. Not a runtime dependency for pokemon-rl itself.

- **[Pokemon Showdown](https://github.com/smogon/pokemon-showdown)** — Node.js battle server. `ShowdownEngine` manages the server process. Battles run locally via WebSocket (no internet connection needed). Requires Node.js.

- **pytest**, **pytest-asyncio** — Testing

No direct numpy, torch, or ML framework dependencies. The env produces text prompts and consumes text responses — the LLM/RL framework is plugged in externally via prime-rl's verifiers.
