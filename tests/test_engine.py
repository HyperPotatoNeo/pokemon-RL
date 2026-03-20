"""Tests for Layer 1: ShowdownEngine.

Unit tests run anywhere. Integration tests need Node.js + Showdown.
"""

import os

import pytest

from pokemon_rl.engine import ShowdownEngine


# ---- Unit tests (no external deps) ----


class TestShowdownEngineUnit:
    @pytest.mark.unit
    def test_init(self):
        engine = ShowdownEngine("/fake/path", port=9999)
        assert engine.port == 9999
        assert not engine.is_running

    @pytest.mark.unit
    def test_health_check_no_server(self):
        engine = ShowdownEngine("/fake/path", port=19999)
        assert not engine.health_check()

    @pytest.mark.unit
    def test_start_missing_path(self):
        engine = ShowdownEngine("/nonexistent/path", port=19999)
        with pytest.raises(FileNotFoundError):
            engine.start()

    @pytest.mark.unit
    def test_repr(self):
        engine = ShowdownEngine("/fake/path", port=9999)
        r = repr(engine)
        assert "9999" in r
        assert "stopped" in r

    @pytest.mark.unit
    def test_context_manager_missing_path(self):
        # Use port 19999 to avoid detecting an existing Showdown server
        with pytest.raises(FileNotFoundError):
            with ShowdownEngine("/nonexistent", port=19999) as engine:
                pass


# ---- Integration tests (need Node.js + Showdown directory) ----


class TestShowdownEngineIntegration:
    @pytest.mark.integration
    def test_start_stop(self, showdown_path, node_path):
        """Start a Showdown server, verify health, stop it."""
        if not os.path.isdir(showdown_path):
            pytest.skip(f"Showdown not found at {showdown_path}")
        if not os.path.isfile(node_path):
            pytest.skip(f"Node.js not found at {node_path}")

        # Use a non-default port to avoid conflicts
        engine = ShowdownEngine(showdown_path, port=8100, node_path=node_path)
        try:
            engine.start(timeout=30)
            assert engine.is_running
            assert engine.health_check()
        finally:
            engine.stop()
            assert not engine.is_running

    @pytest.mark.integration
    def test_external_server_detection(self, showdown_port):
        """If Showdown is already running, engine should detect it."""
        from tests.conftest import HAS_SHOWDOWN

        if not HAS_SHOWDOWN:
            pytest.skip("No Showdown server running for external detection test")

        engine = ShowdownEngine("/unused", port=showdown_port)
        engine.start()  # Should detect external server
        assert engine._externally_managed
        assert engine.health_check()
        engine.stop()  # Should not kill external server
