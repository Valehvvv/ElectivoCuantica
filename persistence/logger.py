"""JsonlEventLogger — append-only event log with configurable fsync.

Design
------
- Each call to ``.log()`` validates the event against the events schema,
  serializes it as a single JSON line, and appends to the file.
- fsync is configurable:
  - For simulation backends: fsync after every event (maximum durability)
  - For hardware backends: batch fsync every N events or every T seconds,
    whichever comes first (reduces I/O overhead when circuits are slow)
- The logger is the SINGLE POINT OF WRITE for events.jsonl. No other
  code should open this file for writing.

Thread safety
-------------
- The logger is NOT thread-safe by default. The heartbeat thread (P3.T3)
  will use its own file handle or acquire the logger's lock.
- A ``threading.Lock`` is provided for external coordination but is NOT
  held during fsync (to avoid blocking the main loop).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from schemas.validators import write_jsonl_line


FsyncMode = Literal["every_write", "batched"]
@dataclass
class BatchConfig:
    """Configuration for batched fsync mode.

    Attributes
    ----------
    max_events
        Force fsync after this many events (even if time limit not reached).
    max_interval_sec
        Force fsync after this many seconds since last fsync.
    """
    max_events: int = 5
    max_interval_sec: float = 30.0
class JsonlEventLogger:
    """Append-only, fsync-aware event logger.

    Parameters
    ----------
    path
        Path to events.jsonl file. Created if it does not exist; parent
        directory must exist.
    mode
        ``"every_write"`` — fsync after every event (default for simulators).
        ``"batched"`` — fsync according to ``batch_config`` (default for
        hardware backends).
    batch_config
        When ``mode="batched"``, controls the fsync cadence.
    lock
        Optional external lock. If None, a new ``threading.Lock`` is created.
    """

    def __init__(
        self,
        path: Path | str,
        mode: FsyncMode = "every_write",
        batch_config: BatchConfig | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        self._path = Path(path)
        self._mode = mode
        self._batch = batch_config or BatchConfig()
        self._lock = lock or threading.Lock()

        # State for batched mode
        self._events_since_fsync = 0
        self._last_fsync_time = time.monotonic()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mode(self) -> FsyncMode:
        return self._mode

    def log(self, event: dict) -> None:
        """Validate ``event`` and append it to the JSONL file.

        The event is written immediately (buffered I/O). fsync happens
        according to the configured mode.
        """
        with self._lock:
            write_jsonl_line(self._path, event)
            self._events_since_fsync += 1

            if self._mode == "every_write":
                self._fsync()
            elif self._mode == "batched":
                elapsed = time.monotonic() - self._last_fsync_time
                if (self._events_since_fsync >= self._batch.max_events or
                        elapsed >= self._batch.max_interval_sec):
                    self._fsync()

    def flush(self) -> None:
        """Force an immediate fsync regardless of mode/batch state."""
        with self._lock:
            self._fsync()

    def _fsync(self) -> None:
        try:
            fd = self._path.open("rb").fileno()  # reuse existing fd
            os.fsync(fd)
        except FileNotFoundError:
            pass  # first write will create the file
        except OSError:
            pass  # fsync may fail on some filesystems (NFS)
        self._events_since_fsync = 0
        self._last_fsync_time = time.monotonic()

    def close(self) -> None:
        """Flush and release resources."""
        self.flush()