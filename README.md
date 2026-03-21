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

## Quick Start

```bash
# On NERSC Perlmutter compute node (see docs/deployment.md):
bash scripts/setup_node.sh

# Run tests:
.venv/bin/python -m pytest -m unit -v          # login node (no Showdown)
bash scripts/run_tests_remote.sh nid008268 -v  # compute node (all tests)
```

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

- **poke-env** (via pokechamp) — Pokemon Showdown client library
- **pokechamp** — Battle state translation with damage calculations
- **Pokemon Showdown** — Node.js battle server
- **pytest**, **pytest-asyncio** — Testing

No direct numpy, torch, or ML framework dependencies. The env produces text prompts and consumes text responses — the LLM/RL framework is plugged in externally via prime-rl's verifiers.
