"""Layer 2a: Player implementations for turn-by-turn battle control.

ControllablePlayer: A poke-env Player whose choose_move() blocks (via
asyncio.Queue) until an external caller provides an action. This inverts
poke-env's callback-driven model into imperative get_state/submit_action.

Opponent factory: Creates opponent players of various types for flexible
matchmaking (heuristic, callback, controllable for self-play).

Concurrency model:
    poke-env runs its own daemon event loop (POKE_LOOP). ControllablePlayer's
    choose_move runs ON POKE_LOOP. External callers (BattleManager, env hooks)
    run on a DIFFERENT event loop. The asyncio.Queue instances are created on
    POKE_LOOP via create_in_poke_loop, and cross-loop access uses
    asyncio.run_coroutine_threadsafe + asyncio.wrap_future.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import random
from typing import Any, Callable, Union

# Atomic counter for unique player names — avoids timestamp collisions
# when creating many concurrent players in the same millisecond.
_player_counter = itertools.count()


def _next_username(prefix: str = "ctrl") -> str:
    """Generate a unique player username using an atomic counter."""
    return f"{prefix}-{next(_player_counter)}-{int(time.time()) % 10000}"


class ControllablePlayer:
    """Player whose choose_move blocks until an external action is submitted.

    This is a factory — call create() to get a poke-env Player instance.
    The Player cannot be instantiated directly because poke-env imports are
    deferred to avoid import-time issues on systems without the full stack.

    Usage (from BattleManager):
        player = ControllablePlayer.create(battle_format="gen1randombattle", ...)
        # In POKE_LOOP: choose_move fires → battle state pushed to state_queue
        # BattleManager reads state_queue, decides action, pushes to action_queue
        # choose_move receives action, returns BattleOrder to poke-env

    Attributes on the returned Player instance:
        state_queue: asyncio.Queue  — battle states pushed by choose_move
        action_queue: asyncio.Queue — actions consumed by choose_move
        finished_event: asyncio.Event — set when battle ends
        result_battle: Battle | None — the finished battle object
    """

    # Timeout for waiting on action_queue.get() inside choose_move.
    # If the external caller crashes, this prevents POKE_LOOP from hanging forever.
    ACTION_TIMEOUT_SECONDS = 300

    @staticmethod
    def create(
        battle_format: str,
        server_config: Any,
        account_name: str | None = None,
        team: str | None = None,
        action_timeout: float = 300,
    ) -> Any:
        """Create a poke-env Player with queue-based external control.

        Args:
            battle_format: e.g. "gen1randombattle"
            server_config: poke-env ServerConfiguration
            account_name: Player username (auto-generated if None)
            team: Team string (None for random battle formats)
            action_timeout: Seconds to wait for action before falling back

        Returns:
            Player instance with state_queue, action_queue, finished_event,
            result_battle attributes.
        """
        from poke_env import AccountConfiguration
        from poke_env.player.player import Player
        from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
        from poke_env.concurrency import create_in_poke_loop

        if account_name is None:
            account_name = _next_username("ctrl")

        class _ControllablePlayerImpl(Player):
            MAX_CONSECUTIVE_TIMEOUTS = 2

            def __init__(self, timeout: float, **kwargs):
                super().__init__(**kwargs)
                self.state_queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue)
                self.action_queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue)
                self.finished_event: asyncio.Event = create_in_poke_loop(asyncio.Event)
                self.result_battle: Any = None
                self._action_timeout = timeout
                self._consecutive_timeouts = 0
                self._max_consecutive_timeouts = self.MAX_CONSECUTIVE_TIMEOUTS

            def _create_forfeit_order(self):
                """Create a forfeit order to end a zombie battle."""
                return DefaultBattleOrder()

            def choose_move(self, battle):
                """Return awaitable — blocks until external action provided."""
                return self._async_choose_move(battle)

            async def _async_choose_move(self, battle):
                """Push battle state out, wait for action in."""
                await self.state_queue.put(battle)
                try:
                    action = await asyncio.wait_for(
                        self.action_queue.get(),
                        timeout=self._action_timeout,
                    )
                    self._consecutive_timeouts = 0
                    return action
                except asyncio.TimeoutError:
                    # Drain any stale action that arrived after timeout.
                    # wait_for cancels the inner get() without consuming,
                    # so a late action would desync all subsequent turns.
                    # Yield first: the pending put was scheduled via
                    # run_coroutine_threadsafe onto POKE_LOOP but hasn't
                    # executed because we haven't yielded. sleep(0) lets
                    # the event loop process it before we drain.
                    await asyncio.sleep(0)
                    while not self.action_queue.empty():
                        try:
                            self.action_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self._consecutive_timeouts += 1
                    if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                        # Force forfeit to prevent zombie battles
                        return self._create_forfeit_order()
                    return self.choose_default_move()

            def _battle_finished_callback(self, battle):
                """Signal game over by putting None sentinel on state queue."""
                self.result_battle = battle
                self.finished_event.set()
                # Put sentinel AFTER setting result so readers can access it
                asyncio.ensure_future(self.state_queue.put(None))

        kwargs = dict(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(account_name, None),
            max_concurrent_battles=1,
        )
        if team is not None:
            kwargs["team"] = team

        return _ControllablePlayerImpl(timeout=action_timeout, **kwargs)


def create_opponent(
    opponent_type: str,
    battle_format: str,
    server_config: Any,
    team: str | None = None,
    callback: Callable | None = None,
    **kwargs,
) -> Any:
    """Factory for creating opponent players.

    Args:
        opponent_type: One of:
            "random" — poke-env RandomPlayer (uniform random legal action)
            "max_damage" — poke-env MaxBasePowerPlayer (highest base power move)
            "callback" — CallbackPlayer with provided callback function
            "controllable" — ControllablePlayer (for self-play or remote LLM)
        battle_format: e.g. "gen1randombattle"
        server_config: poke-env ServerConfiguration
        team: Team string (None for random battle formats)
        callback: Required for "callback" type — fn(battle) -> BattleOrder
        **kwargs: Passed to the player constructor

    Returns:
        Player instance ready to battle.
    """
    from poke_env import AccountConfiguration
    from poke_env.player.random_player import RandomPlayer

    opp_name = _next_username("opp")
    base_kwargs = dict(
        battle_format=battle_format,
        server_configuration=server_config,
        account_configuration=AccountConfiguration(opp_name, None),
        max_concurrent_battles=1,
    )
    if team is not None:
        base_kwargs["team"] = team

    if opponent_type == "random":
        return RandomPlayer(**base_kwargs)

    elif opponent_type == "max_damage":
        from poke_env.player.baselines import MaxBasePowerPlayer
        return MaxBasePowerPlayer(**base_kwargs)

    elif opponent_type == "callback":
        if callback is None:
            raise ValueError("callback opponent_type requires a callback function")
        from pokemon_rl.adapter import CallbackPlayer
        return CallbackPlayer.create(
            callback=callback,
            battle_format=battle_format,
            server_config=server_config,
            account_name=opp_name,
            team=team,
        )

    elif opponent_type == "abyssal":
        from poke_env.player.baselines import AbyssalPlayer
        return AbyssalPlayer(**base_kwargs)

    elif opponent_type == "controllable":
        return ControllablePlayer.create(
            battle_format=battle_format,
            server_config=server_config,
            account_name=_next_username("sp"),  # self-play prefix
            team=team,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown opponent_type: {opponent_type}. "
            f"Expected: random, max_damage, abyssal, callback, controllable"
        )
