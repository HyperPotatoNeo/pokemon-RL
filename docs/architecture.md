# Architecture

pokemon-rl uses a 4-layer architecture. Each layer depends only on layers below it. poke-env imports are deferred (inside functions, not at module level) so layers 3-4 can be imported on systems without poke-env for unit testing.

## Layer Diagram

```
                    +-----------------------+
                    |  prime-rl orchestrator |  (external — calls hooks)
                    +-----------+-----------+
                                |
                    +-----------v-----------+
              L4    |   PokemonBattleEnv    |  env.py
                    |  setup_state          |  4 hooks matching MultiTurnEnv
                    |  get_prompt_messages   |  Configurable rewards
                    |  add_trajectory_step   |  Parse failure tracking
                    |  render_completion     |  Step-level reward callback
                    +-----------+-----------+
                                |
                    +-----------v-----------+
              L3    |   StateTranslator     |  translator.py
                    |  battle_to_prompt     |  "simple" or "pokechamp_io"
                    |  parse_action         |  JSON extraction + matching
                    |  get_fallback_action   |  Random legal action
                    +-----------+-----------+
                                |
              +-----------------+------------------+
              |                                    |
    +---------v----------+            +------------v-----------+
L2  |   BattleManager    | battle.py  |   BattleAdapter        | adapter.py
    |  start_battle      |            |   run_battle           |
    |  step              |            |   (full-battle mode)   |
    |  start_battle_     |            +------------------------+
    |    selfplay        |
    |  submit_selfplay_  |            +------------------------+
    |    action          |            |  ControllablePlayer    | players.py
    |  get_pending_      |            |  create_opponent       |
    |    selfplay_states |            +------------------------+
    |  close             |
    +--------+-----------+
             |
    +--------v-----------+
L1  |   ShowdownEngine   |  engine.py
    |  start / stop      |  Manages Node.js process
    |  health_check      |  Port detection
    +--------------------+
```

## Data Flow: Heuristic Mode

One player controlled externally, opponent auto-responds.

```
1. setup_state
   ├─ Creates BattleManager
   ├─ BattleManager.start_battle(opponent_type="random")
   │   ├─ Creates ControllablePlayer (our player)
   │   ├─ Creates RandomPlayer (opponent)
   │   ├─ Schedules battle_against on POKE_LOOP
   │   └─ Returns first Battle state from state_queue
   └─ Stores battle + manager in state dict

2. get_prompt_messages (loop)
   ├─ Checks game_over, max_game_turns
   └─ StateTranslator.battle_to_prompt(battle) → [system, user] messages

3. add_trajectory_step (loop)
   ├─ StateTranslator.parse_action(completion_text, battle) → BattleOrder
   ├─ If parse fails: get_fallback_action (random), set parse_failed=True
   ├─ BattleManager.step(action) → (next_battle, done)
   ├─ step_reward_fn(battle_before, next_battle, action, 0) → step_reward
   └─ Appends step to trajectory

4. render_completion
   ├─ _assign_rewards(trajectory, won) using reward_win/loss/draw
   └─ Writes metrics dict to state
```

## Data Flow: Self-Play Mode

Both players controlled externally. See [selfplay.md](selfplay.md) for details.

```
1. setup_state
   ├─ BattleManager.start_battle_selfplay()
   │   ├─ Creates 2 ControllablePlayers
   │   ├─ Starts relay tasks on POKE_LOOP
   │   └─ Returns [(0, battle1), (1, battle2)]
   └─ Stores first player's battle, buffers pending states

2-3. get_prompt → add_trajectory_step (alternating players)
     ├─ Hook processes current_player's action
     ├─ _advance_selfplay buffers remaining players
     ├─ After all buffered actions submitted → get_pending_selfplay_states
     └─ New pending states arrive for next turn

4. render_completion
   ├─ P1 wins: P1 steps get reward_win, P2 steps get reward_loss
   ├─ P2 wins: P1 steps get reward_loss, P2 steps get reward_win
   └─ Draw/crash: both get reward_draw
```

## File Map

| File | Layer | Lines | Purpose |
|------|-------|-------|---------|
| `src/pokemon_rl/engine.py` | L1 | ~165 | ShowdownEngine: start/stop Node.js process, health check, port detection. Uses subprocess.DEVNULL (not PIPE) to prevent buffer deadlock. atexit registration for cleanup. |
| `src/pokemon_rl/adapter.py` | L2 | ~230 | BattleAdapter (full-battle via `battle_against`), CallbackPlayer (trajectory capture), `default_action`, `random_action` helpers. |
| `src/pokemon_rl/players.py` | L2 | ~215 | ControllablePlayer (queue-based choose_move), `create_opponent` factory. Atomic username counter. Zombie loop prevention (consecutive timeout → forfeit). |
| `src/pokemon_rl/battle.py` | L2 | ~440 | BattleManager: turn-by-turn orchestration, self-play relay, cross-loop bridge, `close()` + async context manager. Exception propagation from battle future. |
| `src/pokemon_rl/translator.py` | L3 | ~260 | StateTranslator: "simple" and "pokechamp_io" prompt formats, `_extract_last_json` (nested-JSON-aware), `parse_action` with format-aware mechanic validation (dynamax/terastallize). Random fallback action. |
| `src/pokemon_rl/env.py` | L4 | ~580 | PokemonBattleEnv: 4 MultiTurnEnv hooks, configurable rewards (`reward_win`/`loss`/`draw`), `step_reward_fn` callback, `_compute_terminal_reward` + `_assign_rewards` (single source of truth), standalone test modes. |
| `src/pokemon_rl/data.py` | util | ~70 | TrajectoryLogger: JSONL append with atomic `os.write` for concurrent safety. |

## Key Design Decisions

1. **Two-harness separation**: BattleManager (game logic) vs PokemonBattleEnv (LLM logic). Non-LLM agents bypass Layer 4 entirely.

2. **Deferred imports**: All poke-env imports are inside functions, not at module level. This lets unit tests run on systems without poke-env installed.

3. **poke-env data symlink**: `poke_env/` symlink in project root points to pokechamp's copy. Required for poke-env's relative-path data file lookups. Gitignored.

4. **Own venv**: pokemon-rl has its own `.venv` to isolate from system/conda environments. `setup_node.sh` installs pokechamp from local path.
