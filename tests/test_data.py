"""Tests for TrajectoryLogger utility.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    - Verify logged data is exactly what was written (not empty, not garbled)
    - Verify separate log calls produce separate lines (not merged)
    - Verify read-back produces the exact same data structure
"""

import json
import os
import pytest


class TestTrajectoryLogger:
    @pytest.mark.unit
    def test_log_battle_creates_file(self, tmp_path):
        """Logging a battle creates the JSONL file."""
        from pokemon_rl.data import TrajectoryLogger

        path = str(tmp_path / "test.jsonl")
        assert not os.path.exists(path), "File should not exist yet"

        logger = TrajectoryLogger(path)
        logger.log_battle({"won": True, "turns": 10, "reward": 1.0})

        assert os.path.exists(path), "File should exist after logging"

    @pytest.mark.unit
    def test_log_battle_content_roundtrip(self, tmp_path):
        """Data written must be exactly readable back."""
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "test.jsonl"))
        original = {"won": True, "turns": 15, "reward": 1.0, "battle_tag": "test-1"}
        logger.log_battle(original)

        battles = logger.read_battles()
        assert len(battles) == 1, f"Expected 1 battle, got {len(battles)}"
        assert battles[0] == original, (
            f"Read-back doesn't match original:\n"
            f"  Written: {original}\n"
            f"  Read: {battles[0]}"
        )

    @pytest.mark.unit
    def test_log_multiple_battles_separate_lines(self, tmp_path):
        """Each battle must be on its own line."""
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "test.jsonl"))
        logger.log_battle({"won": True, "id": 1})
        logger.log_battle({"won": False, "id": 2})
        logger.log_battle({"won": True, "id": 3})

        battles = logger.read_battles()
        assert len(battles) == 3, f"Expected 3 battles, got {len(battles)}"

        # Verify each is distinct and in order
        assert battles[0]["id"] == 1
        assert battles[1]["id"] == 2
        assert battles[2]["id"] == 3
        assert battles[0]["won"] is True
        assert battles[1]["won"] is False  # This one is different!
        assert battles[2]["won"] is True

    @pytest.mark.unit
    def test_log_step(self, tmp_path):
        """log_step writes a single step line."""
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "steps.jsonl"))
        logger.log_step({"turn": 1, "action": "thunderbolt", "player_idx": 0})
        logger.log_step({"turn": 2, "action": "switch pikachu", "player_idx": 0})

        with open(str(tmp_path / "steps.jsonl")) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

        step1 = json.loads(lines[0])
        step2 = json.loads(lines[1])
        assert step1["turn"] == 1
        assert step2["turn"] == 2
        assert step1["action"] != step2["action"], "Steps should have different actions"

    @pytest.mark.unit
    def test_read_empty_file(self, tmp_path):
        """Reading a non-existent file returns empty list."""
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "nonexistent.jsonl"))
        battles = logger.read_battles()
        assert battles == [], f"Expected empty list, got {battles}"

    @pytest.mark.unit
    def test_log_battle_with_trajectory(self, tmp_path):
        """Verify trajectory list inside battle is preserved."""
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "test.jsonl"))
        original = {
            "won": True,
            "trajectory": [
                {"turn": 1, "action": "move thunderbolt", "reward": 1.0},
                {"turn": 2, "action": "move surf", "reward": 1.0},
            ],
        }
        logger.log_battle(original)

        battles = logger.read_battles()
        assert len(battles[0]["trajectory"]) == 2, "Trajectory should have 2 steps"
        assert battles[0]["trajectory"][0]["action"] == "move thunderbolt"
        assert battles[0]["trajectory"][1]["action"] == "move surf"

    @pytest.mark.unit
    def test_concurrent_writes_atomic(self, tmp_path):
        """H5 fix: concurrent writes should not interleave."""
        import threading
        from pokemon_rl.data import TrajectoryLogger

        logger = TrajectoryLogger(str(tmp_path / "concurrent.jsonl"))
        barrier = threading.Barrier(4)

        def write_batch(thread_id):
            barrier.wait()
            for i in range(10):
                logger.log_step({"thread": thread_id, "step": i})

        threads = [
            threading.Thread(target=write_batch, args=(t,)) for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every line should be valid JSON (no interleaving)
        battles = logger.read_battles()
        assert len(battles) == 40, f"Expected 40 entries, got {len(battles)}"
        for b in battles:
            assert "thread" in b
            assert "step" in b
