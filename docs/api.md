# API Reference

Public classes and methods for pokemon-rl. All poke-env imports are deferred inside methods.

---

## ShowdownEngine (`engine.py`)

Manages a local Pokemon Showdown Node.js server process.

### Constructor

```python
ShowdownEngine(
    showdown_path: str,     # Path to pokemon-showdown directory
    port: int = 8000,       # Server port
    node_path: str = "node" # Path to Node.js binary
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `start(timeout=30)` | `None` | Start server. Blocks until ready. Detects external servers. |
| `stop()` | `None` | Terminate server. No-op if externally managed. |
| `health_check()` | `bool` | True if server accepting connections. |
| `is_running` | `bool` | Property. True if process active or external. |

### Context Manager

```python
with ShowdownEngine("/path/to/showdown", port=8100) as engine:
    assert engine.health_check()
# Automatically stopped on exit
```

---

## BattleAdapter (`adapter.py`)

Full-battle mode via poke-env's `battle_against()`. Runs a complete battle with callback-driven move selection.

### Constructor

```python
BattleAdapter(
    port: int = 8000,
    battle_format: str = "gen1randombattle",
    server_host: str = "localhost"
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `async run_battle(action_fn=None, opponent_type="random", player_team=None, opponent_team=None)` | `dict` | Run complete battle. Returns `{trajectory, won, turns, format, battle_tag}`. |

### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `default_action(battle)` | `Battle -> BattleOrder` | First available move, else first switch. |
| `random_action(battle)` | `Battle -> BattleOrder` | Uniform random legal action. |

---

## ControllablePlayer (`players.py`)

Queue-based Player that blocks `choose_move` until external action provided.

### Factory

```python
player = ControllablePlayer.create(
    battle_format: str,       # e.g. "gen1randombattle"
    server_config: Any,       # poke-env ServerConfiguration
    account_name: str = None, # Auto-generated if None
    team: str = None,         # Team string (None for random)
    action_timeout: float = 300  # Seconds before fallback
)
```

### Attributes on Returned Player

| Attribute | Type | Description |
|-----------|------|-------------|
| `state_queue` | `asyncio.Queue` | Battle states pushed by choose_move (on POKE_LOOP) |
| `action_queue` | `asyncio.Queue` | Actions consumed by choose_move (on POKE_LOOP) |
| `finished_event` | `asyncio.Event` | Set when battle ends |
| `result_battle` | `Battle \| None` | Finished battle object |

### Opponent Factory

```python
opponent = create_opponent(
    opponent_type: str,    # "random", "max_damage", "callback", "controllable"
    battle_format: str,
    server_config: Any,
    team: str = None,
    callback: Callable = None  # Required for "callback" type
)
```

---

## BattleManager (`battle.py`)

Turn-by-turn battle orchestration. Bridges caller's event loop and POKE_LOOP.

### Constructor

```python
BattleManager(
    port: int = 8000,
    battle_format: str = "gen1randombattle",
    server_host: str = "localhost"  # Use node hostname for cross-node
)
```

### Heuristic Mode

| Method | Returns | Description |
|--------|---------|-------------|
| `async start_battle(opponent_type="random", player_team=None, opponent_team=None)` | `Battle \| None` | Start battle, return first state. |
| `async step(action)` | `(Battle \| None, bool)` | Submit action, get `(next_state, done)`. |

### Self-Play Mode

| Method | Returns | Description |
|--------|---------|-------------|
| `async start_battle_selfplay(player1_team=None, player2_team=None)` | `list[(int, Battle)]` | Start with 2 ControllablePlayers. Returns pending states. |
| `async submit_selfplay_action(player_idx, action)` | `None` | Submit action for one player. |
| `async get_pending_selfplay_states()` | `list[(int, Battle)]` | Get next batch. Empty = game over. |

### Results & Lifecycle

| Method | Returns | Description |
|--------|---------|-------------|
| `get_result()` | `dict` | `{won, turns, steps, format, battle_tag, selfplay}` |
| `async close()` | `None` | Cancel futures, relay tasks, clear references. Idempotent. |
| `is_started` | `bool` | Property. |
| `is_finished` | `bool` | Property. |
| `is_selfplay` | `bool` | Property. |

### Async Context Manager

```python
async with BattleManager(port=8000) as mgr:
    battle = await mgr.start_battle()
    # ... play ...
# Automatically cleaned up
```

---

## StateTranslator (`translator.py`)

Converts between poke-env Battle objects and LLM text.

### Constructor

```python
StateTranslator(
    format_style: str = "pokechamp_io"  # "pokechamp_io" or "simple"
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `battle_to_prompt(battle)` | `list[dict]` | `[{role: "system", content: ...}, {role: "user", content: ...}]` |
| `parse_action(response_text, battle)` | `BattleOrder \| None` | Extract last JSON, match against available moves/switches. |
| `get_fallback_action(battle)` | `BattleOrder` | Random legal action (not max-power). |

### Format Styles

| Style | Dependencies | Content |
|-------|-------------|---------|
| `"simple"` | None | Active pokemon, HP, moves (name/power/type), switches. |
| `"pokechamp_io"` | pokechamp, poke-env | Full state with damage calcs via `LocalSim` + `state_translate`. |

### JSON Extraction

`_extract_last_json(text)` (static method) handles nested JSON by scanning backwards from each `}` and trying progressively larger substrings. Returns the last valid dict or `None`.

### Mechanic Validation

`parse_action` validates dynamax/terastallize against the battle format:
- `"dynamax"` key only accepted in gen8 formats
- `"terastallize"` key only accepted in gen9 formats
- Regular `"move"` key works in all formats

---

## PokemonBattleEnv (`env.py`)

LLM harness implementing the MultiTurnEnv 4-hook interface.

### Constructor

```python
PokemonBattleEnv(
    # Components
    adapter: BattleAdapter = None,      # For full_battle mode
    translator: StateTranslator = None, # Prompt/action conversion

    # Mode selection
    control_mode: str = "full_battle",  # "full_battle" or "turn_by_turn"
    opponent_mode: str = "heuristic",   # "heuristic" or "self_play"
    opponent_type: str = "random",      # For heuristic: "random", "max_damage"

    # Game settings
    max_game_turns: int = 200,
    port: int = 8000,
    battle_format: str = "gen1randombattle",
    server_host: str = "localhost",

    # Rewards (see docs/rewards.md)
    reward_win: float = 1.0,
    reward_loss: float = 0.0,
    reward_draw: float = 0.0,
    step_reward_fn: Callable = None,
)
```

### MultiTurnEnv Hooks

| Hook | Signature | Description |
|------|-----------|-------------|
| `async setup_state(state: dict)` | `-> dict` | Initialize battle. Cleans up previous manager. |
| `async get_prompt_messages(state: dict)` | `-> list[dict] \| None` | Return LLM prompt or None (game over). |
| `async add_trajectory_step(state: dict, step: dict)` | `-> None` | Parse action, advance game, record step + step_reward. |
| `async render_completion(state: dict)` | `-> None` | Assign terminal rewards to all trajectory steps. Write metrics. |

### State Dict Keys

Set by `setup_state`:

| Key | Type | Description |
|-----|------|-------------|
| `trajectory` | `list[dict]` | Trajectory steps |
| `game_over` | `bool` | True when game ended |
| `turn` | `int` | Current game turn |
| `decision_count` | `int` | Total decisions (including force-switches) |
| `truncated` | `bool` | True if ended by max_game_turns |
| `won` | `bool \| None` | True/False/None (P1 perspective) |
| `parse_failure_count` | `int` | Unparseable LLM outputs |
| `battle` | `Battle \| None` | Current poke-env Battle object |
| `manager` | `BattleManager \| None` | For turn_by_turn mode |
| `current_player` | `int` | 0 or 1 (self-play only) |
| `_pending_states` | `list[(int, Battle)]` | Buffered self-play states |

### Trajectory Step Keys

Set by `add_trajectory_step`:

| Key | Type | Description |
|-----|------|-------------|
| `completion` | `str` | Raw LLM output (set by caller) |
| `parsed_action` | `str` | poke-env command string |
| `parse_failed` | `bool` | True if fallback was used |
| `player_idx` | `int` | 0 (heuristic) or 0/1 (self-play) |
| `force_switch` | `bool` | True if this was a forced switch |
| `game_turn` | `int` | Battle turn number |
| `step_reward` | `float` | Per-step reward (0.0 if no step_reward_fn) |
| `reward` | `float` | Terminal reward (set by render_completion) |

### Standalone Testing Methods

| Method | Description |
|--------|-------------|
| `async run_standalone(action_fn=None)` | Full battle via BattleAdapter. |
| `async run_turn_by_turn(action_fn=None)` | Step-by-step via BattleManager. Uses `async with` for cleanup. |

### Reward Methods (internal)

| Method | Description |
|--------|-------------|
| `_compute_terminal_reward(won)` | Maps won → reward using config. |
| `_assign_rewards(trajectory, won)` | Assigns terminal rewards to all steps. Handles self-play. |

---

## TrajectoryLogger (`data.py`)

Append-only JSONL logger for battle trajectories. Uses atomic `os.write` for concurrent multi-process safety.

### Constructor

```python
TrajectoryLogger(output_path: str)
```

### Methods

| Method | Description |
|--------|-------------|
| `log_battle(result: dict)` | Write one battle result as JSONL line. |
| `log_step(step: dict)` | Write one turn step as JSONL line. |
| `read_battles()` | Read all entries back. Returns `list[dict]`. |

---

## Module-Level Functions

| Function | Module | Description |
|----------|--------|-------------|
| `_passthrough_reward(state, **kwargs)` | `env.py` | Returns `state["reward"]`. For verifiers rubric passthrough. |
| `_next_username(prefix="ctrl")` | `players.py` | Atomic counter username: `f"{prefix}-{counter}-{time}"`. |
| `default_action(battle)` | `adapter.py` | First available move, else first switch. |
| `random_action(battle)` | `adapter.py` | Uniform random legal action. |
