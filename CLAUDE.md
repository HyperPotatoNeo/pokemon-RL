# pokemon-rl — Development Reference

Pokemon Showdown multi-agent RL environment for prime-rl.

## Architecture (4 Layers)

```
Layer 4: PokemonBattleEnv  — MultiTurnEnv skeleton (verifiers interface)
Layer 3: StateTranslator   — Battle state <-> LLM text (pokechamp format)
Layer 2: BattleAdapter     — Bridges poke-env into imperative control
Layer 1: ShowdownEngine    — Manages Node.js Showdown process
```

## Key Files

| File | Purpose |
|------|---------|
| `src/pokemon_rl/engine.py` | ShowdownEngine: start/stop/health_check |
| `src/pokemon_rl/adapter.py` | BattleAdapter + CallbackPlayer |
| `src/pokemon_rl/translator.py` | StateTranslator (simple + pokechamp_io) |
| `src/pokemon_rl/env.py` | PokemonBattleEnv (MultiTurnEnv hooks) |
| `tests/conftest.py` | Fixtures, capability detection, skip markers |
| `scripts/` | Cluster-specific (gitignored, see scripts/README.md) |

## Dependencies

The project has its own `.venv`. Setup installs:
1. **pokechamp** from local path — brings poke_env, torch, and all transitive deps
2. **pokemon-rl** itself — numpy + test deps

```bash
# On compute node (see scripts/setup_node.sh):
uv pip install -e /path/to/pokechamp   # brings poke-env + deps
uv pip install -e ".[test]"            # our package
```

## Test Markers

- `@pytest.mark.unit` — No external deps. Runs anywhere with the venv active.
- `@pytest.mark.integration` — Needs Showdown server + poke-env.

## Running Tests

```bash
# On compute node with container:
bash scripts/run_tests.sh -m unit         # Unit tests
bash scripts/run_tests.sh -m integration  # Integration tests
bash scripts/run_tests.sh                 # All tests
```

## Current Status

See `PROGRESS.md` for latest updates and `TODO.md` for roadmap.
