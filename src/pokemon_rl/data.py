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

        Args:
            result: Battle result dict (from BattleManager.get_result() or
                PokemonBattleEnv.run_standalone()). Should contain at least:
                won, turns, trajectory, reward, battle_tag.
        """
        with open(self.output_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    def log_step(self, step: dict) -> None:
        """Log a single turn step as one JSONL line.

        Useful for streaming logging during long battles.

        Args:
            step: Turn step dict with keys like: turn, action, player_idx,
                prompt_length, etc.
        """
        with open(self.output_path, "a") as f:
            f.write(json.dumps(step, default=str) + "\n")

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
