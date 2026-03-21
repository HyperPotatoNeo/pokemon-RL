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
| `extract_completion_text(messages)` | `str` | **Static.** Extract text from completion (handles both string and Messages format). Returns last assistant message content from Messages list. |
| `extract_user_content(messages)` | `str` | **Static.** Extract user content from a Messages list. Returns user message content or empty string. |

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

LLM harness implementing the verifiers `MultiTurnEnv` hook interface. Extends `vf.MultiTurnEnv` when verifiers is installed; degrades to plain `object` base for standalone use.

### Constructor

```python
PokemonBattleEnv(
    battle_format: str = "gen1randombattle",
    port: int = 8000,
    server_host: str = "localhost",
    play_mode: str = "single",              # "single" or "self_play"
    opponent_type: str = "random",           # For single mode: "random", "max_damage"
    observation_format: str = "pokechamp_io",# "pokechamp_io" or "simple"
    system_prompt: str | None = None,        # Custom system prompt (None = use translator's)
    reward_win: float = 1.0,                 # Terminal reward for wins
    reward_loss: float = 0.0,                # Terminal reward for losses
    reward_draw: float = 0.0,                # Terminal reward for draws/truncations
    step_reward_fn: Callable | None = None,  # (battle_before, battle_after, action, agent_idx) -> float
    max_game_turns: int = 200,
    num_battles: int = 1000,                 # Dataset size (battle placeholders)
    **kwargs,                                # Forwarded to vf.MultiTurnEnv (minus score_rollouts)
)
```

When verifiers is installed, the constructor creates a `PokemonRubric` and a placeholder HuggingFace `Dataset` of size `num_battles`. `score_rollouts` is forced to `True` (kwargs override prevented).

### Hooks (7 methods)

| Hook | Decorator | Signature | Description |
|------|-----------|-----------|-------------|
| `setup_state` | — | `async (state: dict) -> dict` | Initialize battle, create `_AgentContext`(s). Cleans up previous manager. |
| `game_over` | `@vf.stop` | `async (state: dict) -> bool` | Stop condition: game ended or max turns reached. |
| `get_prompt_messages` | — | `async (state: dict) -> list[dict] \| None` | Build prompt for current agent via `_build_agent_prompt`. |
| `add_trajectory_step` | — | `async (state: dict, step: dict) -> None` | Parse action, advance game, record step metadata in `extras`. |
| `render_completion` | — | `async (state: dict) -> None` | Assign rewards/advantages via `_assign_rewards`. Set metrics. |
| `cleanup_battle` | `@vf.cleanup` | `async (state: dict) -> None` | Close BattleManager on any exit path. Must not raise. |
| `env_response` | — | `async (messages, state) -> list` | Required abstract stub. Unused (overridden by `get_prompt_messages`). |

### Extensible Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `_build_agent_prompt(agent, state)` | `(_AgentContext, dict) -> list[dict]` | Build prompt for current turn. Override for episodic or windowed modes. |

### State Dict Keys

Set by `setup_state`:

| Key | Type | Description |
|-----|------|-------------|
| `trajectory` | `list[dict]` | Trajectory steps |
| `game_over` | `bool` | True when game ended |
| `game_turn` | `int` | Current game turn |
| `truncated` | `bool` | True if ended by max_game_turns |
| `won` | `bool \| None` | True/False/None (P0 perspective) |
| `_agents` | `list[_AgentContext]` | Per-agent state (1 for single, 2 for self-play) |
| `_current_agent_idx` | `int` | Index into `_agents` for next decision |
| `_pending_states` | `list[(int, Battle)]` | Buffered self-play states |
| `manager` | `BattleManager \| None` | Turn-by-turn manager |
| `completion` | `list[dict]` | Messages format (set by `render_completion`) |

Battle objects are accessed via `state["_agents"][idx].battle`, not directly on state.

### Trajectory Step Keys

Set by `add_trajectory_step`. All per-step metadata goes in the `extras` dict:

| Key | Type | Description |
|-----|------|-------------|
| `completion` | `list[dict]` | Messages format (list of role/content dicts), not plain string. Set by verifiers caller. |
| `prompt` | `list[dict]` | Prompt messages (set by verifiers caller) |
| `reward` | `float` | Terminal reward (set by `render_completion` via `_assign_rewards`) |
| `advantage` | `float \| None` | Pre-set for non-uniform rewards (self-play); `None` otherwise |

**`extras` dict keys** (set by `add_trajectory_step`):

| Key | Type | Description |
|-----|------|-------------|
| `agent_idx` | `int` | 0 (single mode) or 0/1 (self-play) |
| `game_turn` | `int` | Battle turn number |
| `force_switch` | `bool` | True if this was a forced switch |
| `parsed_action` | `str` | poke-env command string |
| `parse_failed` | `bool` | True if fallback was used |
| `step_reward` | `float` | Per-step reward (only present if `step_reward_fn` set) |

### Standalone Testing Methods

| Method | Description |
|--------|-------------|
| `async run_standalone(adapter=None, action_fn=None)` | Full battle via BattleAdapter. Requires `adapter` arg. |
| `async run_turn_by_turn(action_fn=None)` | Step-by-step via BattleManager. Uses `async with` for cleanup. Supports self-play. |

### Reward Methods (internal)

| Method | Signature | Description |
|--------|-----------|-------------|
| `_compute_terminal_reward(won)` | `(bool \| None) -> float` | Maps won/loss/draw to reward using config. |
| `_assign_rewards(state)` | `(dict) -> None` | Assigns per-step `reward` and `advantage` from game outcome. Sets `advantage` only when rewards are non-uniform (self-play with a winner); leaves `None` for uniform cases so `score_group` fills cross-rollout normalized values. |

---

## PokemonRubric (`env.py`)

Passthrough reward rubric + Pokemon-specific metrics. Extends `vf.Rubric` when verifiers is installed.

Prevents `score_group` from overwriting env-computed rewards. Metrics are registered explicitly in `__init__` (framework does not auto-discover methods).

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `_passthrough_reward_sync(state)` | `(dict) -> float` | Synchronous passthrough. Returns `state["reward"]` or `0.0`. |
| `passthrough_reward(state)` | `async (dict) -> float` | Async wrapper registered via `add_reward_func`. |
| `won(state)` | `async (dict) -> int` | Metric: `int(state["won"])` or `-1` if None. |
| `game_turns(state)` | `async (dict) -> int` | Metric: `state["game_turn"]`. |
| `parse_failures(state)` | `async (dict) -> int` | Metric: count of `extras["parse_failed"]` across trajectory. |

---

## _AgentContext (`env.py`)

Per-agent state dataclass during a rollout. Data only, no behavior.

```python
@dataclass
class _AgentContext:
    agent_idx: int                       # 0 or 1
    battle: Any = None                   # Current poke-env Battle object
    steps: list = field(default_factory=list)           # This agent's trajectory steps
    message_history: list = field(default_factory=list) # Conversation history (for future episodic mode)
    parse_failure_count: int = 0
    force_switch_count: int = 0
```

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
| `load_environment(**kwargs)` | `__init__.py` | Verifiers env discovery entry point. Verifiers calls `importlib.import_module(env_id)` then `module.load_environment(**env_args)`. Returns `PokemonBattleEnv(**kwargs)`. |
| `_passthrough_reward(state, **kwargs)` | `env.py` | Returns `state["reward"]`. Backward-compat module-level function; canonical path is `PokemonRubric.passthrough_reward`. |
| `_next_username(prefix="ctrl")` | `players.py` | Atomic counter username: `f"{prefix}-{counter}-{time}"`. |
| `default_action(battle)` | `adapter.py` | First available move, else first switch. |
| `random_action(battle)` | `adapter.py` | Uniform random legal action. |
