# Progress

## 2026-03-21: Post-Review Bug Fixes — 10 bugs fixed, 4 test gaps fixed

### Done
- **10 bugs fixed** from adversarial code review (verified against prime-rl source):
  - HIGH: step_reward_fn output folded into step["reward"] (extras dropped at IPC)
  - HIGH: choose_move timeout desync — sleep(0) yield before queue drain
  - MEDIUM: stale next_battle in self-play set to None (matches documented contract)
  - MEDIUM: pokechamp system prompt preserved (system_prompt=None default)
  - MEDIUM: move/switch name normalization — re.sub strips all non-alphanumeric
  - MEDIUM: close() sentinel on selfplay_relay before nulling
  - LOW: dynamax_disable correct for all gens, game_turn monotonic guard,
    render_completion reward fallback, multimodal content block handling
- **4 test gaps fixed**: unconditional truncation assertions (max_game_turns=1),
  cleanup exception suppression test, FailStartManager.close_called assert,
  GPU parse success assertion
- **2 deferred**: queue hang on POKE_LOOP death (rare), Zoroark illusion crash (vendor)
- **Cross-node fix**: poke-env's `websocket_url_online` used `wss://` (SSL) for
  non-localhost servers, hanging on TLS handshake. Fixed in pokechamp fork:
  `vendor/pokechamp/poke_env/ps_client/ps_client.py:376` → `ws://`.
  Cross-node battles now work natively with `BattleManager(server_host="nidXXXXXX")`.

### Test results (2-node cross-node verified)
- Unit: 217 passed
- Integration: 43 passed (real Showdown battles)
- GPU: 6 passed (Qwen3-4B via vLLM + cross-node multinode)
- Total: **266 passed, 0 skipped, 0 failed**

## 2026-03-21: Phase 4 Verifiers Integration — IMPLEMENTED

### Done
- **PokemonBattleEnv(vf.MultiTurnEnv)**: Full hook overrides (setup_state, get_prompt_messages,
  add_trajectory_step, render_completion), conditional verifiers inheritance
- **PokemonRubric**: Passthrough reward + game metrics (won, game_turns, parse_failures),
  explicitly registered via add_reward_func/add_metric (C13)
- **_AgentContext**: Passive dataclass, per-agent state (battle, steps, message_history)
- **_build_agent_prompt**: Fresh prompt mode via translator, custom system_prompt override
- **_assign_rewards(state)**: Per-step rewards, config-derived advantage baseline
  `(reward_win + reward_loss) / 2` for self-play (not step-count-dependent)
- **@vf.stop game_over**: Counts game turns not trajectory steps
- **@vf.cleanup cleanup_battle**: Exception-safe, idempotent
- **Error boundaries**: All BattleManager/translator calls wrapped → vf.Error
- **load_environment()** in __init__.py for verifiers env discovery
- **StateTranslator**: extract_completion_text (Messages→str), extract_user_content,
  _RobustOrder for mock-safe fallback actions
- **ServerConfiguration fix**: uses plain `host:port` format (pokechamp's poke-env fork prepends ws:// internally)
- **Old tests updated**: test_env.py, test_hooks.py — all Phase 4 renames applied
- **3 rounds adversarial review**: cleanup exception safety, extras update vs overwrite,
  _current_agent_idx always set, content=None handling, advantage baseline fix,
  step_reward_fn self-play fix, trajectory retry reset, score_rollouts bypass prevention,
  state["reward"] P0 perspective consistency

### Test results (with verifiers 0.1.9.post3)
- Unit: 207 passed, 0 failed, 0 skipped
- Integration: 28 passed, 1 skipped (pokechamp), 0 failed
- Total: **235 passed**

### Key design decisions
- `reward_win=1.0, reward_loss=0.0, reward_draw=0.0` — wins=1, everything else=0
- Self-play advantage baseline = `(reward_win + reward_loss) / 2`, not within-rollout mean
- `score_rollouts=True` enforced (kwargs.pop prevents bypass)
- `max_turns=-1` disables framework step counting; @vf.stop game_over uses game turns
- Metrics set in render_completion as fallback for standalone mode; PokemonRubric provides
  them in full verifiers pipeline via score_group

## 2026-03-20: Verifiers integration plan complete — 3 rounds adversarial review

### Done
- **Full design plan**: `PHASE_VERIFIERS_PLAN.md` — 13 parts covering architecture,
  hooks, self-play, rewards, dataset, registration, tests, implementation order.
- **3 rounds adversarial review** against actual verifiers source code:
  - Round 1 (correctness): Found `@final rollout()` constraint, score_group advantage
    poisoning, non-existent `vf.register_environment`, wasted LLM call on battle=None.
  - Round 2 (simplicity): Merged 3-method render_completion into single `_assign_rewards`
    override. Moved `extract_completion_text` to StateTranslator. Made `_AgentContext`
    passive dataclass. Added score_group survival test.
  - Round 3 (consistency): Found PokemonRubric methods not auto-registered (silent zero
    rewards — blocking). Fixed passthrough_reward None handling. Fixed setup_state
    double-close. Wrapped translator calls in error boundary.

### Key design decisions
- **Hooks-only**: Works within `@final rollout()` loop. Five overrides + `@vf.stop` + `@vf.cleanup`.
- **Fresh prompts via `_build_agent_prompt`**: Single override point, extensible for
  episodic/windowed modes. `_AgentContext.message_history` always recorded as enabler.
- **Single `_assign_rewards` override**: Handles rewards AND advantages. Auto-detects
  non-uniform rewards (self-play, shaped) and pre-sets advantages to prevent score_group
  from assigning uniform state-level values.
- **`PokemonRubric`**: Combines passthrough reward + game metrics. Methods must be
  explicitly registered via `add_reward_func`/`add_metric` (framework does not auto-discover).
- **play_mode="single"/"self_play"**: Replaces opponent_mode="heuristic"/"self_play".
- **Branching trajectory strategy mandatory** for fresh-prompt mode.

### Verified data flow (reward/advantage pipeline)
```
render_completion → _assign_rewards sets step["reward"] + step["advantage"]
→ score_group: skips pre-set values (only sets if None)
→ extract_result: copies reward, advantage, extras
→ branch_rollout: TrainingSample.reward/advantage from step
→ orchestrator: skips pre-set advantages (only sets if None)
```

## 2026-03-20: Code review fixes — 137 tests passing (111 unit + 26 integration)

### Done
- **All 4 Critical fixes**: Fallback action reward hacking (C1), truncated game rewards (C2),
  draw/crash indistinguishable from loss (C3), dead code cleanup (C4).
- **All 7 High fixes**: PIPE deadlock (H1), BattleManager cleanup (H2), zombie loop prevention (H3),
  name collisions (H4), concurrent-safe logging (H5), socket leak (H6), setup_state cleanup (H7).
- **10 of 13 Medium fixes**: server_host consistency (M1), _format warning (M5), nested JSON parsing (M6),
  gen1 dynamax validation (M7), deferred test imports (M8), numpy removal (M9), event loop blocking (M10),
  exception propagation (M12), relay error handling (M13), grace period (M3 partial).
- **5 of 8 Low fixes**: deprecated asyncio (L3), gpu marker removal (L5), atexit registration (L6),
  PIPE handle cleanup (L7, moot after H1).
- **17 new tests** covering all critical/high fixes.
- **Deferred**: M2 (asymmetric policies), M4 (verifiers registration), M11 (shaped rewards) — design
  decisions, not bugs. L1 (adapter redundancy), L4 (monkey-patch), L8 (counter persistence) — minor.
- Full documentation in `CODE_FIXES.md`.

## 2026-03-20: Adversarial review + selfplay hooks fixes — 120 tests passing

### Done
- **Adversarial code review**: Found 6 bugs in the selfplay hooks path (verifiers integration).
- **5 fixes in `env.py`**:
  - `setup_state` selfplay: unpacked `(idx, battle)` tuples into bare Battle objects
  - `setup_state` selfplay: None check now inspects actual battle values, not tuples
  - `_advance_selfplay`: buffers pending states, calls `get_pending` only after ALL
    buffered actions submitted (prevents deadlock when hooks submit one action at a time)
  - `render_completion` selfplay: `won=None` gives symmetric 0.0 to both players
  - `_run_selfplay_standalone`: same `won=None` fix applied to standalone path
- **2 fixes in `battle.py`**:
  - `get_pending_selfplay_states`: deprecated `get_event_loop()` → `get_running_loop()`
  - Sentinel in grace window: valid state no longer discarded when second item is None
- **Stale comments cleaned**: adapter.py "planned" → documented as implemented in battle.py
- **37 new tests (120 total: 93 unit + 27 integration):**
  - `test_hooks.py` (24 unit): StrictMockSelfplayManager enforces BattleManager contract,
    full hooks cycle for both heuristic and selfplay, setup_state type verification,
    standalone selfplay contract validation
  - `test_env.py` (+5): render_completion edge cases (won=None, empty trajectory),
    hooks heuristic integration, trajectory monotonicity
  - `test_battle.py` (+4): result schema, sentinel handling, concurrent selfplay
  - `test_translator.py` (+3): nested JSON, extra keys, empty moves
  - `test_players.py` (+1): controllable factory queue attributes

### Key design: selfplay hooks buffering
The MultiTurnEnv hooks model calls add_trajectory_step once per LLM response (one player
at a time). But selfplay needs ALL pending actions before Showdown resolves the turn.
Solution: `_advance_selfplay` maintains `_pending_states` buffer. After P1 acts, it pops
P2 from the buffer and sets P2 as current_player. Only after P2 also acts (buffer empty)
does it call `get_pending_selfplay_states()` for the next turn.

## 2026-03-20: Turn-by-turn control + self-play — 84 tests passing

### Done
- **ControllablePlayer** (`players.py`): Queue-based Player that blocks choose_move
  until external action is provided. Uses asyncio.Queue on POKE_LOOP with 300s timeout
  fallback. Game-over sentinel via `_battle_finished_callback` hook.
- **BattleManager** (`battle.py`): Turn-by-turn orchestration bridging caller's event
  loop and poke-env's POKE_LOOP. Heuristic mode: start → step → step → get_result.
  Self-play mode: sequential API handling force-switch asymmetry via thread-safe relay queue.
- **Self-play**: Two ControllablePlayers, relay tasks forward states to shared queue.
  `get_pending_selfplay_states()` handles normal turns (2 states) and force-switches (1 state).
  Opposite reward assignment (P1 wins → P1=1.0, P2=0.0).
- **Enhanced PokemonBattleEnv** (`env.py`): Two control modes (full_battle, turn_by_turn),
  two opponent modes (heuristic, self_play). `run_turn_by_turn()` for testing without LLM.
- **TrajectoryLogger** (`data.py`): Append-only JSONL writer for battle data collection.
- **Opponent factory** (`players.py`): random, max_damage, callback, controllable types.
- **84 tests (61 unit + 23 integration), all passing:**
  - Queue mechanics, state machine guards, sentinel detection, timeout behavior
  - Reward correctness (win≠loss), selfplay opposite rewards, force-switch handling
  - Full game lifecycle: heuristic + selfplay modes, concurrent battles, trajectory integrity

### Key decisions
- **Thread-safe relay queue for self-play**: asyncio.Queue lives on POKE_LOOP but callers
  are on a different loop. Used threading.Queue as relay — POKE_LOOP tasks write,
  caller reads via run_in_executor. Avoids cross-loop deadlocks.
- **Sequential selfplay API**: `get_pending_selfplay_states()` instead of symmetric
  `step_selfplay(p1, p2)`. Force-switches only affect one player — symmetric API
  deadlocks when gather(get_p1, get_p2) but only p1 has a state.
- **Atomic username counter**: `itertools.count()` instead of timestamp-based names.
  Prevents collisions when creating 64+ concurrent players in the same millisecond.
- **No fall-through tests**: Every test verifies both positive AND negative cases.
  Wrong inputs produce different results than correct ones.

### Bug fixed
- **Self-play deadlock**: Original design used `asyncio.gather` to collect both players'
  states simultaneously. When a pokemon faints, only that player gets choose_move —
  the other player's get() hangs forever. Fixed with sequential state collection and
  thread-safe relay queue.

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
- **poke-env via pokechamp**: Installing pokechamp (`pip install -e`) puts poke_env into
  site-packages. No symlinks needed.
- **Circular import fix**: `import poke_env` before `pokechamp.prompts` to avoid
  `pokechamp.prompts → poke_env → baselines → pokechamp.prompts` circular dependency.
- **Real poke-env types in tests**: Mock objects use `Move('id', gen=1)` and
  `Pokemon.__new__(Pokemon)` so `BattleOrder.message` isinstance checks work.
