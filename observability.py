"""Structured logging & per-run result traceability (stdlib only).

Provides:

- Console logging with ANSI colour on the levelname only (TTY-aware; no
  colour codes when ``stderr`` is redirected/piped).
- File logging as JSON Lines (one JSON object per record, no colour),
  written to ``logs/`` (gitignored) so raw logs never pollute the repo.
- ``RunContext``: groups a single execution's outputs (results, plots,
  predictions, metadata) under ``results/<backend_mode>/<run_id>/`` so
  runs can be compared per backend/architecture without clobbering each
  other.

This module is imported by both the main Python >=3.12 environment and
the isolated Python 3.9 ``.venv-spinq`` environment used for the SpinQ
NMR backend (see ``spinq_backend.py``), so it MUST only use the standard
library (plus ``pandas``, which is available in both environments).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent
LOGS_DIR: Path = PROJECT_ROOT / "logs"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Console formatter — ANSI colour on the levelname only, TTY-aware
# ---------------------------------------------------------------------------
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter: ``HH:MM:SS | LEVEL | logger | msg``.

    Colours only the levelname, and only when the destination stream is a
    TTY (so redirected/piped output stays plain, colour-code free).
    """

    def __init__(self, use_color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        levelname = record.levelname
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            levelname = f"{color}{levelname}{_RESET}" if color else levelname
        message = record.getMessage()
        line = f"{ts} | {levelname} | {record.name} | {message}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


# ---------------------------------------------------------------------------
# File formatter — JSON Lines, never coloured
# ---------------------------------------------------------------------------
_RESERVED_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime"}


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per log line; includes any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key in payload:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Metadata redaction
# ---------------------------------------------------------------------------
_SENSITIVE_MARKERS = ("password", "token", "secret", "pass")


def _redact(value: Any) -> Any:
    """Recursively redact dict values whose key looks like a credential."""
    if isinstance(value, dict):
        redacted = {}
        for key, val in value.items():
            key_lower = str(key).lower()
            if any(marker in key_lower for marker in _SENSITIVE_MARKERS):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact(val)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------
class RunContext:
    """Groups a single execution's logger and per-run result directory.

    Attributes
    ----------
    logger : logging.Logger
    run_dir : Path
        ``results/<backend_mode>/<run_id>/`` — created on init.
    run_id : str
    backend_mode : str
    """

    def __init__(
        self,
        logger: logging.Logger,
        run_dir: Path,
        run_id: str,
        backend_mode: str,
    ) -> None:
        self.logger = logger
        self.run_dir = run_dir
        self.run_id = run_id
        self.backend_mode = backend_mode

    def write_metadata(self, meta: dict) -> None:
        """Write ``run_metadata.json`` with credential fields redacted."""
        safe_meta = _redact(meta)
        path = self.run_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(safe_meta, fh, indent=2, default=str)

    def save_predictions(self, df) -> None:
        """Save a predictions DataFrame to ``predictions.csv`` (no index)."""
        df.to_csv(self.run_dir / "predictions.csv", index=False)

    def results_path(self, filename: str) -> Path:
        """Return ``run_dir / filename``."""
        return self.run_dir / filename


# ---------------------------------------------------------------------------
# init_run
# ---------------------------------------------------------------------------
def init_run(backend_mode: str, level: int = logging.INFO) -> RunContext:
    """Configure logging (console + JSON file) and a per-run result dir.

    Parameters
    ----------
    backend_mode : str
        Backend identifier used to namespace ``results/<backend_mode>/``.
    level : int
        Console/logger level (default ``logging.INFO``). The file handler
        always captures ``DEBUG`` and above, regardless of ``level``, so
        detailed diagnostics remain available for later analysis even
        when the console is kept terse.

    Returns
    -------
    RunContext
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RESULTS_DIR / backend_mode / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vqc")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter(use_color=sys.stderr.isatty()))
    logger.addHandler(console_handler)

    log_file = LOGS_DIR / f"{run_id}_{backend_mode}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(file_handler)

    run = RunContext(
        logger=logger,
        run_dir=run_dir,
        run_id=run_id,
        backend_mode=backend_mode,
    )
    logger.info("run iniciado", extra={"run_id": run_id, "backend": backend_mode})
    return run


def git_commit_hash() -> str | None:
    """Return the short git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
