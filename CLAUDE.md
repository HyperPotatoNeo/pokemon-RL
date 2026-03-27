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
| `src/pokemon_rl/eval/` | Eval package: config, llm_player, runner, report |
| `tests/conftest.py` | Fixtures, capability detection, skip markers |
| `scripts/` | Generic scripts (setup, test, launch, eval); cluster-specific in `local_scripts/` |
| `configs/pokemon/` | RL training + eval TOML configs |

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
- `@pytest.mark.gpu` — Needs Showdown + vLLM servers (GPU compute node).

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
| `docs/eval_testing_protocol.md` | Eval testing tiers, commands, debugging playbook |
| `docs/api.md` | API reference for all public classes and methods |

## Eval

Standalone eval framework in `src/pokemon_rl/eval/`. Runs trained agents against
heuristic, metamon, or LLM opponents. See `docs/eval_testing_protocol.md` for the
full testing protocol and `scripts/launch_eval.sh` for the generic launcher.

```bash
# Generic launch (inside container with venv active):
bash scripts/launch_eval.sh configs/pokemon/eval_example.toml

# NERSC production (sbatch):
sbatch local_scripts/launch_eval_prod.sh configs/pokemon/eval_example.toml
```

Key files: `src/pokemon_rl/eval/{config,llm_player,runner,report}.py`,
`tests/test_eval_{unit,integration,gpu}.py`, `configs/pokemon/eval_example.toml`.

## Current Status

Phase 5 RL training + eval feature complete. Interleaved trajectory feature complete and validated.
400 tests passing (352 existing + 48 interleaved). Production run: 4 steps completed on 2-node setup with interleaved mode (entropy declining, seq lengths 17-19.5K).

Key details:
- pokechamp_io prompts include CoT constraint: forces 3-sentence reasoning inside JSON
- `trajectory_strategy`: `"branching"` (each turn = separate sample) or `"interleaved"` (full conversation = 1 sample)
- Interleaved: two-phase per turn (reasoning 512 tokens + extraction 50 tokens), lighter prompts after turn 1
- `rollouts_per_example = batch_size` — batch-level GRPO normalization
- `[inference.server] port = 8001` — must differ from Showdown port 8000
- `[trainer.model.ac] freq = 1` — gradient checkpointing prevents OOM on single GPU
- AdamW `betas1=0.9, betas2=0.9` — all production configs
- Configs in `configs/pokemon/`, generic launch in `scripts/launch_rl.sh`
- Cross-node: requires `--net=host` containers (pokechamp fork patched ws:// for non-localhost)

Performance optimizations (cumulative):
- `_copy_battle=False` in LocalSim: eliminates deepcopy, **11x speedup** (69 min/step to 6.4 min/step)
- Mutation fix in `vendor/pokechamp/pokechamp/prompts.py`: local `max_hp` var instead of `pokemon._max_hp = 1`
- `asyncio.to_thread` in `get_prompt_messages`: considered but not yet implemented
- Metrics: `wins`, `losses`, `draws` logged separately to W&B

Multi-node (2-node):
- Node 1: inference DP=4 (standalone, `configs/pokemon/inference_node1.toml`)
- Node 2: inference DP=2 + trainer (2 GPUs) + Showdown + orchestrator
- Launch: `sbatch local_scripts/launch_2node_prod.sh configs/pokemon/rl_vs_abyssal_600_4x4.toml`
- Step time: ~6-7 min with batch_size=128

See `docs/rl_training.md` for the full training guide.
