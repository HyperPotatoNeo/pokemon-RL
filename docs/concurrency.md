# Concurrency Model

pokemon-rl bridges two separate event loops. Understanding this is essential for debugging hangs, deadlocks, and resource leaks.

## The Two Event Loops

```
Thread 1 (caller):     asyncio event loop (pytest, orchestrator, etc.)
Thread 2 (poke-env):   POKE_LOOP — daemon thread, runs poke-env's WebSocket client
```

poke-env creates `POKE_LOOP` at import time as a daemon thread. All poke-env operations (battle_against, choose_move, WebSocket I/O) run on this loop. The caller's code runs on a different loop.

**The asyncio.Queue instances** (state_queue, action_queue) live on POKE_LOOP. They are created via `create_in_poke_loop(asyncio.Queue)` which schedules creation on POKE_LOOP so the queue is bound to that loop.

## Cross-Loop Bridge

`BattleManager._poke_loop_get` and `_poke_loop_put` (`battle.py:108-136`) handle the bridge:

```python
# Reading from a POKE_LOOP queue from the caller's loop:
async def _poke_loop_get(self, queue):
    from poke_env.concurrency import POKE_LOOP
    current_loop = asyncio.get_running_loop()
    if current_loop is POKE_LOOP:
        return await queue.get()         # Same loop — direct await
    else:
        future = asyncio.run_coroutine_threadsafe(queue.get(), POKE_LOOP)
        return await asyncio.wrap_future(future)  # Cross-loop — bridge
```

`asyncio.run_coroutine_threadsafe` schedules the coroutine on POKE_LOOP and returns a `concurrent.futures.Future`. `asyncio.wrap_future` makes that awaitable on the caller's loop.

## Self-Play Relay Pattern

Self-play adds a third concurrency layer. Two ControllablePlayers produce states on POKE_LOOP, but the caller needs a single queue to poll.

```
POKE_LOOP:
  Player1.state_queue ──> relay_task_0 ──┐
                                          ├──> thread_queue.Queue (relay)
  Player2.state_queue ──> relay_task_1 ──┘

Caller's loop:
  get_pending_selfplay_states() <── run_in_executor(relay.get)
```

**Why thread_queue.Queue?** asyncio.Queue is bound to one loop. We need POKE_LOOP tasks to write and the caller's loop to read. `threading.Queue` (thread-safe, loop-agnostic) bridges both.

**Relay tasks** (`battle.py:265-277`) are async coroutines running on POKE_LOOP:

```python
async def relay_states(player_idx, player, relay_q):
    try:
        while True:
            state = await player.state_queue.get()
            relay_q.put((player_idx, state))  # thread-safe, non-blocking
            if state is None:
                break
    except asyncio.CancelledError:
        relay_q.put((player_idx, None))  # Sentinel on cancel
    except Exception as exc:
        relay_q.put((player_idx, None))  # Sentinel on error
```

**Critical**: `relay_q` is captured as a **local parameter**, not accessed via `self._selfplay_relay`. This prevents a race condition where `close()` sets `self._selfplay_relay = None` while the CancelledError handler tries to call `.put()` on it.

## Grace Period for State Collection

`get_pending_selfplay_states()` (`battle.py:310-361`) collects states from the relay:

1. **First read**: `run_in_executor(relay.get, timeout=300)` — blocks up to 5 min
2. **Grace period**: `run_in_executor(relay.get, timeout=0.5)` — checks for second player

Both reads are wrapped in `run_in_executor` to avoid blocking the caller's event loop. Normal turns produce both states near-simultaneously; force-switches produce only one.

## ControllablePlayer Lifecycle

```
1. create()           — builds Player subclass with queues on POKE_LOOP
2. battle_against()   — scheduled on POKE_LOOP, starts WebSocket battle
3. choose_move()      — called by poke-env ON POKE_LOOP
   ├─ state_queue.put(battle)    — push state to caller
   ├─ action_queue.get(timeout)  — wait for caller's action
   │   ├─ Success: return action, reset timeout counter
   │   ├─ Timeout: consecutive_timeouts++
   │   │   ├─ < MAX: return choose_default_move()
   │   │   └─ >= MAX: return DefaultBattleOrder() (forfeit)
   └─ Return BattleOrder to poke-env
4. _battle_finished_callback()  — sets finished_event, puts None sentinel
```

The **zombie loop prevention** (`players.py:110-124`) ensures that if the caller crashes, the player doesn't hang for 300s × N turns. After 2 consecutive timeouts (10 minutes), it forfeits.

## Resource Cleanup

`BattleManager.close()` (`battle.py:398-419`):
1. Cancels `_battle_future` (stops the battle coroutine on POKE_LOOP)
2. Cancels all relay tasks
3. Clears player references (allows GC of WebSocket connections)
4. Sets `_selfplay_relay = None` and `_finished = True`

Callers should use `async with BattleManager(...) as mgr:` for automatic cleanup. The standalone paths (`run_turn_by_turn`, `_run_selfplay_standalone`) use this pattern.

## Common Pitfalls

1. **Never `await` a POKE_LOOP queue directly from the caller's loop** — use `_poke_loop_get`/`_poke_loop_put` which detect the current loop and bridge if needed.

2. **Relay tasks run on POKE_LOOP, not the caller's loop** — they are scheduled via `run_coroutine_threadsafe`, not `asyncio.create_task`.

3. **`close()` is async** because it may need to propagate through the event loop. Always `await` it.

4. **Sentinel detection**: `None` on state_queue means game over. Non-None means a real Battle state. The relay preserves this — `(idx, None)` means game over for that player.
