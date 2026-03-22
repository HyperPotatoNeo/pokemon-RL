"""Shared test fixtures and markers.

Test categories:
    @pytest.mark.unit         — No external deps. Runs anywhere.
    @pytest.mark.integration  — Needs Showdown server + poke-env (compute node).

Dependencies (poke-env, pokechamp) are installed via the project venv.
See scripts/setup_node.sh for setup instructions.
"""

import os
import socket

import pytest


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
def _is_port_open(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            return sock.connect_ex(("localhost", port)) == 0
    except Exception:
        return False


try:
    from poke_env.player.player import Player  # noqa: F401

    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False

try:
    from pokechamp.prompts import state_translate  # noqa: F401

    HAS_POKECHAMP = True
except ImportError:
    HAS_POKECHAMP = False

SHOWDOWN_PORT = int(os.environ.get("SHOWDOWN_PORT", "8000"))
HAS_SHOWDOWN = _is_port_open(SHOWDOWN_PORT)

# Skip markers
requires_poke_env = pytest.mark.skipif(
    not HAS_POKE_ENV, reason="poke-env not installed (run scripts/setup_node.sh)"
)
requires_pokechamp = pytest.mark.skipif(
    not HAS_POKECHAMP, reason="pokechamp not installed (run scripts/setup_node.sh)"
)
requires_showdown = pytest.mark.skipif(
    not HAS_SHOWDOWN, reason=f"Showdown server not running on port {SHOWDOWN_PORT}"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def showdown_path():
    # Default: vendor/pokechamp/pokemon-showdown inside the repo
    default = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "vendor", "pokechamp", "pokemon-showdown",
    )
    return os.environ.get("SHOWDOWN_PATH", default)


@pytest.fixture
def node_path():
    return os.environ.get("NODE_PATH", "node")


@pytest.fixture
def showdown_port():
    return SHOWDOWN_PORT
