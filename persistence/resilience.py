"""Resilience utilities: signal handler, watchdog, heartbeat.

Design
------
Signal handler uses the self-pipe pattern with O_NONBLOCK for reliable
signal delivery even during blocking operations.

The watchdog is phase-aware:
  - simulation (statevector/aer):  300s timeout
  - spinq_nmr:                    1800s timeout  
  - ibm:                          3600s timeout

Heartbeat logs a ``heartbeat`` event to the logger every 30 seconds
from a daemon thread. The signal handler sets a shared ``shutdown_flag``
that the heartbeat and main loop check periodically.
"""

from __future__ import annotations

import fcntl
import os
import select
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from persistence.logger import JsonlEventLogger


# ---------------------------------------------------------------------------
# Phase-aware timeout config
# ---------------------------------------------------------------------------

BACKEND_TIMEOUTS: dict[str, float] = {
    "statevector": 300.0,
    "aer": 300.0,
    "spinq_nmr": 1800.0,
}


def get_watchdog_timeout(backend: str, default: float = 3600.0) -> float:
    """Return the watchdog timeout in seconds for the given backend."""
    return BACKEND_TIMEOUTS.get(backend, default)


# ---------------------------------------------------------------------------
# Self-pipe signal handler (O_NONBLOCK)
# ---------------------------------------------------------------------------

class SelfPipeSignal:
    """Self-pipe trick with O_NONBLOCK for reliable signal delivery.

    Usage::

        with SelfPipeSignal(signal.SIGINT) as sp:
            sp.wait(timeout=1.0)  # returns True if signal caught
            # or use sp.fileno() in a select() call
    """

    def __init__(self, signum: int = signal.SIGINT) -> None:
        self.signum = signum
        self._r_fd: int | None = None
        self._w_fd: int | None = None
        self._old_handler: Any = None

    def __enter__(self) -> SelfPipeSignal:
        r_fd, w_fd = os.pipe()
        # Set read end to non-blocking
        flags = fcntl.fcntl(r_fd, fcntl.F_GETFL)
        fcntl.fcntl(r_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        # Set write end to non-blocking too
        flags = fcntl.fcntl(w_fd, fcntl.F_GETFL)
        fcntl.fcntl(w_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        self._r_fd = r_fd
        self._w_fd = w_fd

        def _handler(signo: int, frame: Any) -> None:
            """Signal handler: write a byte to the pipe."""
            try:
                os.write(self._w_fd, b"x")
            except BlockingIOError:
                pass  # pipe full, signal already pending
            except OSError:
                pass

        self._old_handler = signal.signal(self.signum, _handler)
        return self

    def __exit__(self, *args: Any) -> None:
        # Restore old signal handler
        signal.signal(self.signum, self._old_handler)
        # Close pipe
        if self._r_fd is not None:
            try:
                os.close(self._r_fd)
            except OSError:
                pass
        if self._w_fd is not None:
            try:
                os.close(self._w_fd)
            except OSError:
                pass
        self._r_fd = None
        self._w_fd = None

    @property
    def fileno(self) -> int:
        """File descriptor for use in select/poll."""
        if self._r_fd is None:
            msg = "SelfPipeSignal not entered (use 'with' block)"
            raise RuntimeError(msg)
        return self._r_fd

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for signal with optional timeout.

        Returns True if signal was caught, False on timeout.
        """
        if self._r_fd is None:
            msg = "SelfPipeSignal not entered (use 'with' block)"
            raise RuntimeError(msg)
        r, _, _ = select.select([self._r_fd], [], [], timeout)
        if r:
            # Drain the pipe
            try:
                os.read(self._r_fd, 4096)
            except OSError:
                pass
            return True
        return False

    def drain(self) -> None:
        """Drain any pending signal bytes without blocking."""
        if self._r_fd is not None:
            try:
                os.read(self._r_fd, 4096)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class Watchdog:
    """Phase-aware watchdog timer.

    Raises TimeoutError if the run exceeds the configured timeout.
    Must be periodically ``.tick()``-ed by the main loop.
    """

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = timeout_sec
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.timeout_sec - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.elapsed >= self.timeout_sec

    def tick(self) -> None:
        """Check timeout; raise TimeoutError if expired."""
        if self.expired:
            msg = (
                f"Watchdog timeout after {self.timeout_sec:.0f}s "
                f"(phase: {self.timeout_sec:.0f}s limit)"
            )
            raise TimeoutError(msg)

    def reset(self) -> None:
        """Reset the timer (e.g. after entering a new phase)."""
        self._start = time.monotonic()


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat:
    """Daemon thread that logs a ``heartbeat`` event every ``interval_sec``.

    The thread checks ``shutdown_flag`` after each write and stops when
    the flag is set.
    """

    def __init__(
        self,
        logger: JsonlEventLogger,
        interval_sec: float = 30.0,
    ) -> None:
        self._logger = logger
        self._interval = interval_sec
        self._thread: threading.Thread | None = None
        self._shutdown_flag = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return  # already started
        self._shutdown_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_flag.is_set()

    def _run(self) -> None:
        while not self._shutdown_flag.wait(self._interval):
            try:
                self._logger.log({
                    "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "level": "info",
                    "run_id": "heartbeat",  # Add run_id for heartbeat events
                    "event": "heartbeat",
                    "data": {"thread": "heartbeat", "interval": self._interval},
                })
            except Exception:
                pass  # heartbeat should never crash


# ---------------------------------------------------------------------------
# Graceful Shutdown Coordinator
# ---------------------------------------------------------------------------

@dataclass
class ShutdownState:
    """Shared state for graceful shutdown coordination."""
    requested: bool = False
    reason: str = ""
    iteration: int = 0
    timestamp: str = ""


class GracefulShutdown:
    """Orchestrate graceful shutdown on signal or timeout.

    Usage::

        shutdown = GracefulShutdown(logger, run_dir, "statevector")
        with SelfPipeSignal(signal.SIGINT) as sp:
            while not shutdown.should_stop:
                if sp.wait(timeout=1.0):
                    shutdown.request("interrupted")
                    break
                if watchdog.expired:
                    shutdown.request("timeout")
                    break
                # ... main loop work ...
    """

    def __init__(
        self,
        logger: JsonlEventLogger,
        run_dir: Path,
        backend: str,
        ansatz: str = "",
    ) -> None:
        self._logger = logger
        self._run_dir = run_dir
        self.backend = backend
        self.ansatz = ansatz
        self.state = ShutdownState()

    @property
    def should_stop(self) -> bool:
        return self.state.requested

    def request(self, reason: str, iteration: int = 0) -> None:
        if self.state.requested:
            return  # already requested
        self.state.requested = True
        self.state.reason = reason
        self.state.iteration = iteration
        self.state.timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        # Extract run_id from run_dir (parent directory name)
        run_id = self._run_dir.name
        self._logger.log({
            "ts": self.state.timestamp,
            "level": "warning",
            "run_id": run_id,
            "event": "shutdown",
            "data": {
                "reason": reason,
                "iteration": iteration,
                "backend": self.backend,
            },
        })
        self._logger.flush()