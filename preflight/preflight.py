"""Pre-flight check: verify backend SDK is available. NO FALLBACK.

The architectural rule for this project is: if the requested backend
cannot be used, the run fails immediately with a clear error event.
Silent degradation to a different backend is forbidden because it
corrupts the experiment (a previous run's results were invalidated
exactly because of this).

This module is called once at the start of every run, BEFORE any data
loading or training. It emits JSONL events to the run's events.jsonl
file and returns a PreflightResult.

Design constraints
------------------
- Pure structural check: imports only, no I/O, no network, no hardware.
  Connection checks happen in T4 (calibration).
- Mock-friendly: each backend has its own ``_check_<backend>`` function
  that tests can override.
- Async-signal-safe: only stdlib imports here. No signal handling.
- Exit code: ``run_preflight`` does NOT call ``sys.exit``. The caller
  (main runner, T7) decides what to do with the result.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Literal

from schemas.validators import validate_event, ValidationError

# Local timezone offset (CLT = UTC-3). For human-readable timestamps.
try:
    _TZ_OFFSET = datetime.now(timezone.utc).astimezone().utcoffset() or timedelta(0)
except Exception:  # pragma: no cover - extremely defensive
    _TZ_OFFSET = timedelta(0)


def _now_iso() -> str:
    """Return current local time as RFC 3339 with timezone offset."""
    dt = datetime.now().astimezone()
    # Format: YYYY-MM-DDTHH:MM:SS.sss±HH:MM
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + dt.strftime("%z")[:3] + ":" + dt.strftime("%z")[3:]


BackendMode = Literal["statevector", "aer_simulator", "spinq_nmr", "ibm_simulator", "ibm_hardware"]


@dataclass
class PreflightResult:
    """Outcome of a pre-flight check.

    Attributes
    ----------
    ok
        True if the requested backend is usable. False otherwise.
    backend
        The requested backend mode (echoes BACKEND_MODE).
    backend_resolved
        For now, same as ``backend``. Future: may report a forced
        different backend if we ever allow non-silent redirection.
        For T3, this is always equal to ``backend`` because we forbid
        fallback.
    sdk_version
        Version string of the SDK, or None if not available.
    error
        Human-readable error description if ``ok`` is False, else None.
    events_path
        Path to the JSONL file where preflight events were written.
    """

    ok: bool
    backend: str
    backend_resolved: str
    sdk_version: str | None
    error: str | None
    events_path: Path

    def to_event_data(self) -> dict:
        return {
            "backend": self.backend,
            "backend_resolved": self.backend_resolved,
            "sdk_version": self.sdk_version,
            "ok": self.ok,
            "error": self.error,
        }


# --- Per-backend checks -------------------------------------------------------

def _check_statevector() -> tuple[bool, str | None]:
    """Qiskit is always required (even statevector uses it)."""
    try:
        import qiskit  # noqa: F401
        return True, getattr(qiskit, "__version__", "unknown")
    except ImportError as exc:
        return False, f"qiskit not importable: {exc}"


def _check_aer() -> tuple[bool, str | None]:
    try:
        import qiskit_aer  # noqa: F401
        return True, getattr(qiskit_aer, "__version__", "unknown")
    except ImportError as exc:
        return False, f"qiskit_aer not importable: {exc}"


def _check_spinq() -> tuple[bool, str | None]:
    """SpinQ NMR: only check SDK importability. Connection in T4."""
    if importlib.util.find_spec("spinqit") is None:
        return False, (
            "spinqit not importable. Required for BACKEND_MODE=spinq_nmr. "
            "Install from the .whl provided by the professor. "
            "See docs/spinq_setup.md."
        )
    try:
        import spinqit  # type: ignore[import-not-found]  # noqa: F401
        return True, getattr(spinqit, "__version__", "unknown")
    except ImportError as exc:
        return False, f"spinqit import failed: {exc}"


def _check_ibm() -> tuple[bool, str | None]:
    try:
        import qiskit_ibm_runtime  # noqa: F401
        return True, getattr(qiskit_ibm_runtime, "__version__", "unknown")
    except ImportError as exc:
        return False, f"qiskit_ibm_runtime not importable: {exc}"


_BACKEND_CHECKS: dict[str, Callable[[], tuple[bool, str | None]]] = {
    "statevector": _check_statevector,
    "aer_simulator": _check_aer,
    "spinq_nmr": _check_spinq,
    "ibm_simulator": _check_ibm,
    "ibm_hardware": _check_ibm,
}


def _resolve_backend(env_var: str = "BACKEND_MODE", default: str = "statevector") -> str:
    return os.environ.get(env_var, default).strip().lower()


# --- Event emission -----------------------------------------------------------

def _emit_event(events_path: Path, run_id: str, event: str, level: str,
                data: dict) -> None:
    payload = {
        "ts": _now_iso(),
        "level": level,
        "run_id": run_id,
        "event": event,
        "data": data,
    }
    try:
        validate_event(payload)
    except ValidationError as exc:
        # Should never happen; if it does, the event would corrupt JSONL.
        # Log to stderr but do not raise (we are inside preflight itself).
        print(f"preflight: internal event validation failed: {exc}", file=sys.stderr)
        return
    with events_path.open("a", encoding="utf-8") as fh:
        import json
        fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


# --- Public API ---------------------------------------------------------------

def run_preflight(
    run_id: str,
    events_path: Path,
    backend: str | None = None,
) -> PreflightResult:
    """Run the pre-flight check and write JSONL events.

    Parameters
    ----------
    run_id
        Timestamp-based run identifier (e.g. ``"20260706_091500"``).
    events_path
        File where preflight events will be appended. Created if it
        does not exist; parent directory must exist.
    backend
        Backend mode to check. If None, reads ``BACKEND_MODE`` env var,
        falling back to ``"statevector"``.

    Returns
    -------
    PreflightResult
        Always returned (never raised). Caller decides whether to
        continue or exit.
    """
    backend = (backend or _resolve_backend()).strip().lower()
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    _emit_event(events_path, run_id, "preflight_start", "info", {"backend": backend})

    if backend not in _BACKEND_CHECKS:
        msg = f"unknown BACKEND_MODE='{backend}'. Valid: {sorted(_BACKEND_CHECKS)}"
        _emit_event(events_path, run_id, "preflight_failed", "error", {
            "backend": backend, "reason": msg,
        })
        return PreflightResult(
            ok=False, backend=backend, backend_resolved=backend,
            sdk_version=None, error=msg, events_path=events_path,
        )

    check = _BACKEND_CHECKS[backend]
    try:
        ok, info = check()
    except Exception as exc:  # noqa: BLE001 - we want to catch any import error
        ok = False
        info = f"unexpected error during {backend} check: {exc!r}"

    if ok:
        _emit_event(events_path, run_id, "preflight_ok", "info", {
            "backend": backend, "sdk_version": info,
        })
        return PreflightResult(
            ok=True, backend=backend, backend_resolved=backend,
            sdk_version=info, error=None, events_path=events_path,
        )

    # Failure: error event + result.ok=False
    _emit_event(events_path, run_id, "preflight_failed", "error", {
        "backend": backend, "reason": info,
    })
    return PreflightResult(
        ok=False, backend=backend, backend_resolved=backend,
        sdk_version=None, error=info, events_path=events_path,
    )