"""Layer 2b: BattleManager — turn-by-turn battle orchestration.

Bridges between an external caller's event loop and poke-env's POKE_LOOP
to provide imperative battle control: start_battle → step → step → get_result.

Supports two modes:
    1. Heuristic opponent: Our player is ControllablePlayer, opponent auto-responds.
    2. Self-play: Both players are ControllablePlayer, caller drives both.

Concurrency:
    poke-env's battle_against() runs on POKE_LOOP (daemon thread).
    BattleManager's async methods run on the caller's event loop (different thread).
    Cross-loop bridge: asyncio.run_coroutine_threadsafe + asyncio.wrap_future.

Self-play force-switch handling:
    In Pokemon, when a pokemon faints only that player gets a new choose_move call.
    The other player has already submitted. This means the number of pending states
    varies per cycle. The selfplay API handles this with a sequential model:
    get_pending_selfplay_states() returns however many states are pending.

Usage (heuristic):
    manager = BattleManager(port=8000, battle_format="gen1randombattle")
    battle = await manager.start_battle(opponent_type="random")
    while battle is not None:
        action = decide(battle)
        battle, done = await manager.step(action)
        if done: break
    result = manager.get_result()

Usage (self-play):
    manager = BattleManager(port=8000, battle_format="gen1randombattle")
    pending = await manager.start_battle_selfplay()
    while pending:
        for idx, state in pending:
            action = decide(state)
            await manager.submit_selfplay_action(idx, action)
        pending = await manager.get_pending_selfplay_states()
    result = manager.get_result()
"""

from __future__ import annotations

import asyncio
import logging
import queue as thread_queue  # thread-safe queue
from typing import Any

logger = logging.getLogger(__name__)


class BattleManager:
    """Manages a single battle with turn-by-turn control.

    One BattleManager per battle. For concurrent battles, create multiple
    BattleManager instances (each creates its own Player pair).

    Args:
        port: Showdown server port (default 8000)
        battle_format: Pokemon Showdown format string
        server_host: Showdown server hostname (default "localhost", use
            node hostname for cross-node play)
    """

    def __init__(
        self,
        port: int = 8000,
        battle_format: str = "gen1randombattle",
        server_host: str = "localhost",
    ):
        self.port = port
        self.battle_format = battle_format
        self.server_host = server_host
        self._server_config = None

        # Players (set during start_battle)
        self._player = None          # ControllablePlayer (always)
        self._opponent = None        # Any Player type
        self._opponent_player2 = None  # ControllablePlayer for self-play

        # Battle task running on POKE_LOOP
        self._battle_future = None

        # State
        self._started = False
        self._finished = False
        self._selfplay = False
        self._step_count = 0

        # Self-play: thread-safe queue for state relay
        # Both players' states get relayed here as (player_idx, battle) tuples
        self._selfplay_relay = None  # thread_queue.Queue
        self._relay_tasks = []

    def _get_server_config(self):
        """Lazy-load server configuration."""
        if self._server_config is None:
            from poke_env.ps_client.server_configuration import ServerConfiguration
            self._server_config = ServerConfiguration(
                f"{self.server_host}:{self.port}",
                "https://play.pokemonshowdown.com/action.php?",
            )
        return self._server_config


    # ------------------------------------------------------------------
    # Cross-event-loop bridge
    # ------------------------------------------------------------------

    async def _poke_loop_get(self, queue: asyncio.Queue) -> Any:
        """Read from an asyncio.Queue that lives on POKE_LOOP."""
        from poke_env.concurrency import POKE_LOOP

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is POKE_LOOP:
            return await queue.get()
        else:
            future = asyncio.run_coroutine_threadsafe(queue.get(), POKE_LOOP)
            return await asyncio.wrap_future(future)

    async def _poke_loop_put(self, queue: asyncio.Queue, item: Any) -> None:
        """Write to an asyncio.Queue that lives on POKE_LOOP."""
        from poke_env.concurrency import POKE_LOOP

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is POKE_LOOP:
            await queue.put(item)
        else:
            future = asyncio.run_coroutine_threadsafe(queue.put(item), POKE_LOOP)
            await asyncio.wrap_future(future)

    # ------------------------------------------------------------------
    # Heuristic opponent mode
    # ------------------------------------------------------------------

    async def start_battle(
        self,
        opponent_type: str = "random",
        player_team: str | None = None,
        opponent_team: str | None = None,
        **opponent_kwargs,
    ) -> Any:
        """Start a battle with a heuristic or callback opponent.

        Returns the first Battle state, or None if battle failed to start.
        """
        if self._started:
            raise RuntimeError("Battle already started. Create a new BattleManager.")

        from pokemon_rl.players import ControllablePlayer, create_opponent
        from poke_env.concurrency import POKE_LOOP

        server_config = self._get_server_config()

        self._player = ControllablePlayer.create(
            battle_format=self.battle_format,
            server_config=server_config,
            team=player_team,
        )


        self._opponent = create_opponent(
            opponent_type=opponent_type,
            battle_format=self.battle_format,
            server_config=server_config,
            team=opponent_team,
            **opponent_kwargs,
        )


        self._battle_future = asyncio.run_coroutine_threadsafe(
            self._player._battle_against(self._opponent, 1),
            POKE_LOOP,
        )
        self._started = True
        self._selfplay = False

        state = await self._poke_loop_get(self._player.state_queue)
        if state is None:
            self._finished = True
        return state

    def _check_battle_future(self) -> None:
        """Check if the battle coroutine raised an exception.

        M12: Propagates exceptions from _battle_future instead of
        letting callers hang on queues that will never be fed.
        """
        if self._battle_future is not None and self._battle_future.done():
            exc = self._battle_future.exception()
            if exc is not None:
                self._finished = True
                raise RuntimeError(
                    f"Battle coroutine failed: {exc}"
                ) from exc

    async def step(self, action: Any) -> tuple[Any, bool]:
        """Submit action and wait for next state. Returns (state, done)."""
        if not self._started:
            raise RuntimeError("Call start_battle() first.")
        if self._finished:
            raise RuntimeError("Battle already finished.")
        if self._selfplay:
            raise RuntimeError("Use step_selfplay methods for self-play battles.")
        self._check_battle_future()

        self._step_count += 1
        await self._poke_loop_put(self._player.action_queue, action)

        state = await self._poke_loop_get(self._player.state_queue)
        if state is None:
            self._finished = True
            return None, True
        return state, False

    # ------------------------------------------------------------------
    # Self-play mode
    #
    # Uses a relay pattern: async tasks on POKE_LOOP read from each
    # player's state_queue and put tagged tuples (player_idx, battle)
    # onto a thread-safe queue. This lets the caller's event loop
    # poll a single queue without cross-loop deadlocks.
    # ------------------------------------------------------------------

    async def start_battle_selfplay(
        self,
        player1_team: str | None = None,
        player2_team: str | None = None,
    ) -> list[tuple[int, Any]]:
        """Start a self-play battle with two ControllablePlayers.

        Returns list of (player_idx, battle_state) for players needing actions.
        On the first turn, both players need actions: [(0, s1), (1, s2)].
        """
        if self._started:
            raise RuntimeError("Battle already started. Create a new BattleManager.")

        from pokemon_rl.players import ControllablePlayer
        from poke_env.concurrency import POKE_LOOP

        server_config = self._get_server_config()

        self._player = ControllablePlayer.create(
            battle_format=self.battle_format,
            server_config=server_config,
            team=player1_team,
        )

        self._opponent_player2 = ControllablePlayer.create(
            battle_format=self.battle_format,
            server_config=server_config,
            team=player2_team,
        )


        # Thread-safe relay queue: POKE_LOOP tasks write, caller reads
        self._selfplay_relay = thread_queue.Queue()

        # Start relay tasks on POKE_LOOP
        # N2 fix: relay_q captured as local parameter so close() can set
        # self._selfplay_relay = None without causing AttributeError in
        # the CancelledError handler on POKE_LOOP.
        async def relay_states(player_idx, player, relay_q):
            """Read from player's state_queue, put on shared relay queue."""
            try:
                while True:
                    state = await player.state_queue.get()
                    relay_q.put((player_idx, state))
                    if state is None:
                        break
            except asyncio.CancelledError:
                relay_q.put((player_idx, None))
            except Exception as exc:
                logger.error(f"Relay task for player {player_idx} failed: {exc}")
                relay_q.put((player_idx, None))

        for idx, player in enumerate([self._player, self._opponent_player2]):
            task = asyncio.run_coroutine_threadsafe(
                relay_states(idx, player, self._selfplay_relay), POKE_LOOP
            )
            self._relay_tasks.append(task)

        # Schedule battle
        self._battle_future = asyncio.run_coroutine_threadsafe(
            self._player._battle_against(self._opponent_player2, 1),
            POKE_LOOP,
        )
        self._started = True
        self._selfplay = True

        # Collect initial states (first turn always has both)
        return await self.get_pending_selfplay_states()

    async def submit_selfplay_action(
        self, player_idx: int, action: Any
    ) -> None:
        """Submit action for one player in self-play mode.

        Call once for each (player_idx, state) from get_pending_selfplay_states().
        """
        if not self._selfplay:
            raise RuntimeError("Not in selfplay mode.")

        player = self._player if player_idx == 0 else self._opponent_player2
        await self._poke_loop_put(player.action_queue, action)
        self._step_count += 1

    async def get_pending_selfplay_states(self) -> list[tuple[int, Any]]:
        """Get the next batch of pending states in self-play mode.

        Handles:
        - Normal turns: both players need actions → [(0,s1), (1,s2)]
        - Force-switches: only fainting player needs action → [(idx, state)]
        - Game over: returns [] with is_finished set to True

        Waits for at least one state, then collects any others that arrive
        within a brief grace period (normal turns produce near-simultaneous
        states).
        """
        if self._finished:
            return []

        self._check_battle_future()
        relay = self._selfplay_relay

        # Wait for at least one state (blocking with a long timeout)
        try:
            first = await asyncio.get_running_loop().run_in_executor(
                None, lambda: relay.get(timeout=300)
            )
        except thread_queue.Empty:
            self._finished = True
            return []

        idx, state = first
        if state is None:
            self._finished = True
            return []

        results = [(idx, state)]

        # Brief grace period: check if the other player also has a state
        # (normal turns produce both states near-simultaneously)
        # M10: Wrapped in run_in_executor to avoid blocking the event loop
        try:
            second = await asyncio.get_running_loop().run_in_executor(
                None, lambda: relay.get(timeout=0.5)
            )
            idx2, state2 = second
            if state2 is None:
                # Game ending — but return the valid state we already have.
                # The caller can process it; next call will return [].
                self._finished = True
                return results
            results.append((idx2, state2))
        except thread_queue.Empty:
            pass  # Only one state pending (force-switch)

        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result(self) -> dict:
        """Get battle result after game ends."""
        if not self._finished:
            raise RuntimeError("Battle not finished yet.")

        battle = self._player.result_battle
        if battle is None:
            battles = self._player.battles
            if battles:
                battle = list(battles.values())[0]

        if battle is None:
            return {
                "won": None, "turns": 0, "steps": self._step_count,
                "format": self.battle_format, "battle_tag": "unknown",
                "selfplay": self._selfplay,
            }

        return {
            "won": battle.won,
            "turns": battle.turn,
            "steps": self._step_count,
            "format": self.battle_format,
            "battle_tag": battle.battle_tag,
            "selfplay": self._selfplay,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Clean up all resources.

        Cancels the battle future, relay tasks, and clears player references.
        Safe to call multiple times.
        """
        # Cancel battle future
        if self._battle_future is not None and not self._battle_future.done():
            self._battle_future.cancel()
        self._battle_future = None

        # Cancel relay tasks
        for task in self._relay_tasks:
            if not task.done():
                task.cancel()
        self._relay_tasks.clear()

        # Unblock any reader waiting on the relay queue before nulling it
        if self._selfplay_relay is not None:
            self._selfplay_relay.put((None, None))

        self._player = None
        self._opponent = None
        self._opponent_player2 = None
        self._selfplay_relay = None
        self._finished = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def is_selfplay(self) -> bool:
        return self._selfplay
