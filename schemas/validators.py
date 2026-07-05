"""Reusable JSON Schema validators for events.jsonl and state.json.

The single source of truth for the event/state schemas lives in
``events.schema.json`` and ``state.schema.json``. This module wraps
``jsonschema`` and exposes a small, ergonomic API for the rest of the
codebase. Validators are compiled once (Draft 2020-12) and cached.

Design constraints
------------------
- Async-signal-safe callers: validation is pure-Python, no I/O. Safe to
call from anywhere (including loss loops and signal handlers' deferred
work, though the latter is discouraged).
- Strict mode: ``additionalProperties=True`` is preserved in the schema,
but unknown top-level keys are warnings, not errors, to ease forward
compatibility. Unknown keys inside ``data`` are always allowed.
- Human-readable errors: every failure raises ``ValidationError`` with
  a one-line message that includes the line number (for JSONL parsing)
  and the failing field path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import jsonschema
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).parent
EVENTS_SCHEMA_PATH = SCHEMAS_DIR / "events.schema.json"
STATE_SCHEMA_PATH = SCHEMAS_DIR / "state.schema.json"
class ValidationError(ValueError):
    """Raised when an event or state object fails schema validation.

    The message includes the field path and a short human description.
    The original ``jsonschema`` exception is available as ``__cause__``.
    """
def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
# Compiled once at import time.
_EVENTS_VALIDATOR = Draft202012Validator(_load_schema(EVENTS_SCHEMA_PATH))
_STATE_VALIDATOR = Draft202012Validator(_load_schema(STATE_SCHEMA_PATH))
def validate_event(event: dict[str, Any], *, line_no: int | None = None) -> dict[str, Any]:
    """Validate a single event dict against the events schema.

    Parameters
    ----------
    event
        The event payload (already parsed from JSONL).
    line_no
        Optional 1-indexed line number in events.jsonl, used to enrich
        the error message if validation fails.

    Returns
    -------
    The same ``event`` dict (returned for chaining convenience).

    Raises
    ------
    ValidationError
        If the event does not match the schema. The message includes
        the field path and, when known, the line number.
    """
    errors = sorted(_EVENTS_VALIDATOR.iter_errors(event), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.absolute_path) or "<root>"
        prefix = f"events.jsonl:{line_no}: " if line_no is not None else ""
        raise ValidationError(f"{prefix}event validation failed at '{path}': {first.message}")
    return event
def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Validate a state.json snapshot against the state schema.

    Returns
    -------
    The same ``state`` dict (returned for chaining convenience).

    Raises
    ------
    ValidationError
        If the state does not match the schema.
    """
    errors = sorted(_STATE_VALIDATOR.iter_errors(state), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ValidationError(f"state validation failed at '{path}': {first.message}")
    return state
def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_no, event)`` pairs from a JSONL file.

    Blank lines and lines containing only whitespace are skipped. Each
    non-blank line is parsed and validated with :func:`validate_event`.
    Lines that fail to parse or validate raise :class:`ValidationError`
    with the line number included.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            validate_event(event, line_no=line_no)
            yield line_no, event
def write_jsonl_line(path: Path, event: dict[str, Any]) -> None:
    """Validate ``event`` and append it as one line to ``path``.

    Does NOT fsync; that is the responsibility of the caller (e.g.
    :class:`JsonlEventLogger` in P2.T1) which decides on batching.
    """
    validate_event(event)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")