"""Layer 1: Pokemon Showdown server management.

Manages the Node.js Pokemon Showdown process. One engine per compute node.
The engine starts the server, provides health checks, and handles clean shutdown.

Usage:
    engine = ShowdownEngine("/path/to/pokemon-showdown")
    engine.start()
    assert engine.health_check()
    # ... run battles ...
    engine.stop()
"""

import os
import socket
import subprocess
import time
from pathlib import Path


class ShowdownEngine:
    """Manages a local Pokemon Showdown Node.js server process.

    Args:
        showdown_path: Path to the pokemon-showdown directory (contains the
            `pokemon-showdown` Node.js entry script).
        port: Port to run the server on. Default 8000 (poke-env default).
        node_path: Path to the Node.js binary. Default "node" (assumes on PATH).
    """

    def __init__(
        self,
        showdown_path: str,
        port: int = 8000,
        node_path: str = "node",
    ):
        self.showdown_path = Path(showdown_path)
        self.port = port
        self.node_path = node_path
        self._process: subprocess.Popen | None = None
        self._externally_managed = False

    def start(self, timeout: float = 30) -> None:
        """Start the Showdown server. Blocks until server is ready.

        If a server is already running on the port, marks it as externally
        managed and returns without starting a new process.

        Raises:
            FileNotFoundError: If pokemon-showdown script not found.
            TimeoutError: If server doesn't become ready within timeout.
        """
        # Check if already running on this port
        if self._is_port_open():
            self._externally_managed = True
            return

        if self._process is not None:
            raise RuntimeError("ShowdownEngine already has a running process")

        entry_script = self.showdown_path / "pokemon-showdown"
        if not entry_script.exists():
            raise FileNotFoundError(
                f"pokemon-showdown script not found at {entry_script}"
            )

        # Start the Node.js process
        env = os.environ.copy()
        # Ensure node is on PATH (Showdown's script calls `node build` internally)
        node_dir = str(Path(self.node_path).parent)
        env["PATH"] = node_dir + ":" + env.get("PATH", "")

        self._process = subprocess.Popen(
            [
                self.node_path,
                "pokemon-showdown",
                "start",
                "--no-security",
                "--port",
                str(self.port),
            ],
            cwd=str(self.showdown_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for server to be ready
        if not self._wait_for_ready(timeout):
            self.stop()
            raise TimeoutError(
                f"Showdown server failed to start within {timeout}s on port {self.port}"
            )

    def _wait_for_ready(self, timeout: float) -> bool:
        """Poll until the server port is accepting connections."""
        start = time.time()
        while time.time() - start < timeout:
            if self._process.poll() is not None:
                # Process exited — read stderr for clues
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"Showdown process exited with code {self._process.returncode}: {stderr[:500]}"
                )
            if self._is_port_open():
                return True
            time.sleep(0.5)
        return False

    def _is_port_open(self) -> bool:
        """Check if the server port is accepting TCP connections."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("localhost", self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def health_check(self) -> bool:
        """Check if the server is accepting connections."""
        return self._is_port_open()

    def stop(self) -> None:
        """Stop the Showdown server.

        Does nothing if the server was externally managed (not started by us).
        """
        if self._externally_managed:
            self._externally_managed = False
            return

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    @property
    def is_running(self) -> bool:
        """True if a server process is active (ours or external)."""
        if self._externally_managed:
            return self._is_port_open()
        return self._process is not None and self._process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        if not self._externally_managed:
            self.stop()

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        managed = " (external)" if self._externally_managed else ""
        return f"ShowdownEngine(port={self.port}, {status}{managed})"
