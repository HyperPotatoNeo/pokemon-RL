"""Utility: Trajectory logging for data collection and analysis.

Writes battle trajectories to JSONL files. Each line is one complete battle
or one turn step, depending on the logging granularity.

Usage:
    logger = TrajectoryLogger("battles.jsonl")
    logger.log_battle(result)  # one line per battle
    logger.log_step(step)      # one line per turn (streaming)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Append-only JSONL logger for battle trajectories.

    Args:
        output_path: Path to the JSONL file. Created if it doesn't exist.
            Parent directories are NOT created automatically.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def log_battle(self, result: dict) -> None:
        """Log a complete battle result as one JSONL line.

        Uses os.write for atomic writes (lines < PIPE_BUF = 4KB on Linux).
        Safe for concurrent multi-process writers to the same file.

        Args:
            result: Battle result dict (from BattleManager.get_result() or
                PokemonBattleEnv.run_standalone()). Should contain at least:
                won, turns, trajectory, reward, battle_tag.
        """
        line = json.dumps(result, default=str) + "\n"
        fd = os.open(
            str(self.output_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def log_step(self, step: dict) -> None:
        """Log a single turn step as one JSONL line.

        Uses os.write for atomic writes. Safe for concurrent writers.

        Args:
            step: Turn step dict with keys like: turn, action, player_idx,
                prompt_length, etc.
        """
        line = json.dumps(step, default=str) + "\n"
        fd = os.open(
            str(self.output_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def read_battles(self) -> list[dict]:
        """Read all logged battles from the JSONL file.

        Returns:
            List of battle result dicts.
        """
        if not self.output_path.exists():
            return []
        battles = []
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    battles.append(json.loads(line))
        return battles
