"""BattleCoordinator — manages concurrent battle throttling.

Sits between the orchestrator and PokemonBattleEnv to control how many
battles run simultaneously within a worker process. This is the right
layer for concurrency control — the env handles game logic, the
coordinator handles scheduling.

Usage in TOML config:
    [orchestrator.env.args]
    max_concurrent_battles = 8   # default

The env passes this to the coordinator. The coordinator provides
acquire/release context management around battle lifecycle.

Usage in code:
    coordinator = BattleCoordinator.get(max_concurrent=8)
    async with coordinator.battle_slot():
        # Start and play a battle
        ...
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

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
        return _instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        global _instance
        _instance = None

    @asynccontextmanager
    async def battle_slot(self):
        """Context manager that acquires a battle slot.

        Blocks until a slot is available (if at max_concurrent).
        Releases the slot on exit (including on exception).

        Usage:
            async with coordinator.battle_slot():
                battle = await manager.start_battle(...)
                # ... play game ...
                await manager.close()
        """
        await self._semaphore.acquire()
        self._active += 1
        self._total += 1
        try:
            yield
        finally:
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
