# Self-Play

Self-play mode runs both players under external control. This enables LLM-vs-LLM training where both sides learn simultaneously via GRPO.

## Pokemon Force-Switch Asymmetry

The core complexity of Pokemon self-play: **when a pokemon faints, only the fainting player gets a new choose_move call**. The other player has already submitted their action for that turn.

This means the number of pending actions per turn cycle is **variable**:
- Normal turn: 2 pending states (both players choose)
- Force-switch: 1 pending state (only the fainting player chooses a replacement)
- Double faint: 2 pending states (both need replacements)

A symmetric API like `step_selfplay(p1_action, p2_action)` would **deadlock** when only one player has a pending state.

## Sequential Selfplay API

BattleManager uses a sequential model (`battle.py:229-392`):

```python
manager = BattleManager(port=8000)
pending = await manager.start_battle_selfplay()
# pending = [(0, battle_p1), (1, battle_p2)]

while pending:
    for idx, state in pending:
        action = decide(state)
        await manager.submit_selfplay_action(idx, action)
    pending = await manager.get_pending_selfplay_states()
    # Normal turn: [(0, s1), (1, s2)]
    # Force-switch: [(0, s1)]  or  [(1, s2)]
    # Game over: []

result = manager.get_result()
```

### How get_pending_selfplay_states Works

1. **Wait for first state**: Blocking read on relay queue (up to 300s timeout)
2. **Grace period**: Non-blocking read with 0.5s timeout for second state
3. **Return**: List of `(player_idx, battle)` tuples

Normal turns produce both states near-simultaneously (both players get choose_move after Showdown resolves). Force-switches produce one. The grace period catches the near-simultaneous case.

## Hooks-Path Buffering

The MultiTurnEnv hooks model calls `add_trajectory_step` **once per LLM response** — one player at a time. But self-play needs ALL pending actions submitted before Showdown resolves the turn.

Solution: `_advance_selfplay` (`env.py:458-496`) maintains a `_pending_states` buffer.

### Flow for a Normal Turn (2 pending states)

```
State: _pending_states = [(0, battle_p1), (1, battle_p2)]
       _current_agent_idx = 0

Hook call #1 (P1's turn):
  1. add_trajectory_step gets completion from LLM
  2. parse_action → BattleOrder
  3. _advance_selfplay(action, player_idx=0):
     a. submit_selfplay_action(0, action)
     b. Remove (0, ...) from _pending_states
     c. _pending_states still has [(1, battle_p2)]
     d. Set _current_agent_idx = 1, battle = battle_p2

Hook call #2 (P2's turn):
  1. get_prompt_messages returns P2's prompt
  2. add_trajectory_step gets P2's completion
  3. _advance_selfplay(action, player_idx=1):
     a. submit_selfplay_action(1, action)
     b. Remove (1, ...) from _pending_states
     c. _pending_states is now EMPTY
     d. Call get_pending_selfplay_states() → new turn
     e. Update _current_agent_idx and battle from new pending
```

### Flow for a Force-Switch (1 pending state)

```
State: _pending_states = [(0, battle_p1)]  # Only P1 faints
       _current_agent_idx = 0

Hook call #1 (P1's force-switch):
  1. add_trajectory_step → parse → _advance_selfplay
  2. submit_selfplay_action(0, action)
  3. _pending_states empty → get_pending_selfplay_states()
  4. New pending states for next turn
```

### Key Invariants

1. `get_pending_selfplay_states()` is ONLY called when `_pending_states` is empty (all buffered actions submitted)
2. `_current_agent_idx` always matches the first entry in `_pending_states`
3. `state["_agents"][idx].battle` is always a bare Battle object, never a `(idx, battle)` tuple

## Relay Queue Architecture

See [concurrency.md](concurrency.md) for the full relay pattern. Key points:

- Two relay tasks on POKE_LOOP forward states to a shared `threading.Queue`
- Relay tasks capture `relay_q` as a local parameter (not `self._selfplay_relay`) to prevent race conditions with `close()`
- Relay tasks put `(player_idx, None)` sentinel on error/cancel

## Reward Assignment

In self-play, rewards are per-player (see [rewards.md](rewards.md)):

```python
if won:      # P1 won (from poke-env's perspective)
    P1 steps → reward_win
    P2 steps → reward_loss
elif won is False:  # P2 won
    P1 steps → reward_loss
    P2 steps → reward_win
elif won is None:   # draw/crash
    both → reward_draw
```

`state["won"]` is from **P1's perspective** (the first ControllablePlayer created). P1 = player_idx 0, P2 = player_idx 1.

## Standalone Testing

`run_turn_by_turn` with `play_mode="self_play"` runs self-play without an LLM:

```python
env = PokemonBattleEnv(
    play_mode="self_play",
    observation_format="simple",
)
result = await env.run_turn_by_turn(action_fn=random_action)
```

`_run_selfplay_standalone` directly calls the BattleManager sequential API (no hooks buffering), acting as a reference implementation. Both players use the same `action_fn`. The manager is automatically cleaned up via `async with`.
