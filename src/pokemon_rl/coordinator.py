"""BattleCoordinator — manages concurrent battle throttling.

Sits between the orchestrator and PokemonBattleEnv to control how many
battles run simultaneously within a worker process. The env handles game
logic; the coordinator handles scheduling.

Usage in TOML config:
    [orchestrator.env.args]
    max_concurrent_battles = 8   # default

The env calls acquire() in setup_state and release() in cleanup_battle.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Singleton coordinator per worker process
_instance: BattleCoordinator | None = None


class BattleCoordinator:
    """Controls concurrent battle scheduling within a worker process.

    Uses an asyncio.Semaphore to limit how many battles run at once.
    Battles exceeding the limit are queued and start when a slot opens.

    This is a process-level singleton — all env instances in the same
    worker share the same coordinator.
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0
        self._total = 0
        logger.info(
            "BattleCoordinator initialized (max_concurrent=%d)", max_concurrent
        )

    @classmethod
    def get(cls, max_concurrent: int = 8) -> BattleCoordinator:
        """Get or create the process-level coordinator singleton."""
        global _instance
        if _instance is None:
            _instance = cls(max_concurrent=max_concurrent)
        elif _instance.max_concurrent != max_concurrent:
            logger.warning(
                "BattleCoordinator already initialized with max_concurrent=%d, "
                "ignoring requested max_concurrent=%d",
                _instance.max_concurrent,
                max_concurrent,
            )
        return _instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        global _instance
        _instance = None

    async def acquire(self) -> None:
        """Acquire a battle slot. Blocks if at max_concurrent."""
        await self._semaphore.acquire()
        self._active += 1
        self._total += 1

    def release(self) -> None:
        """Release a battle slot."""
        self._active -= 1
        self._semaphore.release()

    @property
    def active_battles(self) -> int:
        """Number of battles currently running."""
        return self._active

    @property
    def total_battles(self) -> int:
        """Total battles started since coordinator creation."""
        return self._total
