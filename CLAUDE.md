# pokemon-rl — Development Reference

Pokemon Showdown multi-agent RL environment for prime-rl.

**Detailed documentation**: See `docs/` folder — start with `docs/architecture.md` for
the 4-layer design, `docs/concurrency.md` for the POKE_LOOP bridging model,
`docs/rewards.md` for the configurable reward system, `docs/selfplay.md` for
self-play mechanics. Full API reference in `docs/api.md`.

## IMPORTANT: Always use the project .venv

```bash
# On login node:
.venv/bin/python -m pytest ...

# On compute node (inside container):
source .venv/bin/activate && python -m pytest ...

# NEVER use system python, conda envs, or other venvs
```

## Architecture: Two Harnesses

**BattleManager** = general battle harness. Knows Pokemon, players, turns, game state.
Does NOT know about LLMs, prompts, tokens, or rewards. Non-LLM agents (RL policy
networks, heuristics, metamon) use this directly.

**PokemonBattleEnv** = LLM harness on top. Translates battle state → text prompts,
parses text → actions, assigns rewards. Any LLM-in-the-loop use goes through here,
whether for RL training (verifiers), eval, data collection, or human text play.

```
Layer 4: PokemonBattleEnv  — LLM harness (verifiers hooks, prompts, rewards)
Layer 3: StateTranslator   — Battle state <-> LLM text (pokechamp format)
Layer 2: BattleManager     — General battle harness (turn-by-turn control)
         BattleAdapter     — Full-battle mode (callback-driven, legacy)
         ControllablePlayer — Queue-based external control
Layer 1: ShowdownEngine    — Manages Node.js Showdown process
```

## Key Files

| File | Purpose |
|------|---------|
| `src/pokemon_rl/engine.py` | ShowdownEngine: start/stop/health_check |
| `src/pokemon_rl/adapter.py` | BattleAdapter + CallbackPlayer (full-battle mode) |
| `src/pokemon_rl/players.py` | ControllablePlayer + opponent factory |
| `src/pokemon_rl/battle.py` | BattleManager (turn-by-turn orchestration) |
| `src/pokemon_rl/translator.py` | StateTranslator (simple + pokechamp_io) |
| `src/pokemon_rl/env.py` | PokemonBattleEnv (MultiTurnEnv hooks) |
| `src/pokemon_rl/data.py` | TrajectoryLogger (JSONL battle logging) |
| `tests/conftest.py` | Fixtures, capability detection, skip markers |
| `scripts/` | Generic scripts (setup, test, launch); cluster-specific in `local_scripts/` |
| `configs/pokemon/` | RL training TOML configs (selfplay, vs_heuristic, test) |

## Dependencies

The project has its own `.venv`. Setup installs:
1. **pokechamp** from submodule (`vendor/pokechamp`) — brings poke_env fork (ws:// fix), torch, and all transitive deps
2. **pokemon-rl** itself — test deps (pytest, pytest-asyncio)

```bash
# On compute node (see scripts/setup_node.sh):
uv pip install -e vendor/pokechamp    # poke-env fork + deps
uv pip install -e ".[test]"           # our package
```

## Test Markers

- `@pytest.mark.unit` — No external deps. Runs anywhere with the venv active.
- `@pytest.mark.integration` — Needs Showdown server + poke-env (compute node).

## Running Tests

```bash
# Unit tests (no Showdown needed):
.venv/bin/python -m pytest -m unit -v

# All tests (Showdown must be running):
bash scripts/run_tests.sh -v
bash scripts/run_tests.sh -m integration -v
```

## Documentation

| Doc | When to read |
|-----|-------------|
| `docs/architecture.md` | Before modifying any source file — understand the 4-layer design |
| `docs/concurrency.md` | Before touching battle.py or players.py — POKE_LOOP bridging |
| `docs/rewards.md` | Before changing reward logic — configurable system, single source of truth |
| `docs/selfplay.md` | Before touching selfplay code — force-switch handling, hooks buffering |
| `docs/testing.md` | Before writing tests — no-fall-through philosophy, mock patterns |
| `docs/rl_training.md` | Before RL training work — configs, prompt construction, launching |
| `docs/deployment.md` | Before running on Perlmutter — containers, Showdown, SLURM |
| `docs/api.md` | API reference for all public classes and methods |

## Current Status

Phase 5 RL training integration complete. 298+ tests passing.
Full pipeline validated: Qwen3-4B self-play GRPO on gen9randombattle, 3 training steps,
balanced ~50/50 win/loss rewards.

Key details:
- pokechamp_io prompts include CoT constraint: forces 3-sentence reasoning inside JSON
- `trajectory_strategy = "branching"` — each turn = separate TrainingSample (CRITICAL)
- `rollouts_per_example = batch_size` — batch-level GRPO normalization
- `[inference.server] port = 8001` — must differ from Showdown port 8000
- `[trainer.model.ac] freq = 1` — gradient checkpointing prevents OOM on single GPU
- Configs in `configs/pokemon/`, generic launch in `scripts/launch_rl.sh`
- Cross-node: requires `--net=host` containers (pokechamp fork patched ws:// for non-localhost)

See `docs/rl_training.md` for the full training guide.
