# Progress

## 2026-03-19: Initial skeleton — all tests passing

### Done
- Created project structure at `$SCRATCH/pokemon-rl/`
- 4-layer architecture implemented:
  - **Layer 1 (ShowdownEngine)**: Start/stop/health_check for Node.js Showdown process.
    Sets PATH for node binary so Showdown's internal `node build` works.
  - **Layer 2 (BattleAdapter)**: Full-battle mode via poke-env's `battle_against()` with
    `CallbackPlayer` for trajectory capture. Supports random and default action functions.
  - **Layer 3 (StateTranslator)**: Two formats — "simple" (minimal, always works) and
    "pokechamp_io" (full damage calcs via pokechamp's `state_translate` + `LocalSim`).
    Action parsing extracts last JSON from LLM response, matches against available actions.
  - **Layer 4 (PokemonBattleEnv)**: MultiTurnEnv skeleton with 4 hooks + `run_standalone()`.
    Passthrough rubric for verifiers integration.
- **37 tests, all passing:**
  - 26 unit tests (action parsing, env state machine, imports)
  - 11 integration tests (battle lifecycle, prompt generation, full game loop, engine start/stop)
- Cluster scripts: allocate, setup_node (container + Showdown + venv), run_tests

### Key decisions
- **Own venv**: pokemon-rl has its own `.venv`. `setup_node.sh` installs pokechamp from local
  path (brings poke_env, torch, and all transitive deps). No PYTHONPATH hacks.
- **poke_env data symlink**: pokechamp's `poke_env` looks up data files via relative paths.
  `poke_env` symlink in project root points to pokechamp's copy (gitignored).
- **Circular import fix**: `import poke_env` before `pokechamp.prompts` to avoid
  `pokechamp.prompts → poke_env → baselines → pokechamp.prompts` circular dependency.
- **Real poke-env types in tests**: Mock objects use `Move('id', gen=1)` and
  `Pokemon.__new__(Pokemon)` so `BattleOrder.message` isinstance checks work.
