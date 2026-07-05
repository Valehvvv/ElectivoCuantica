"""StateSnapshot — derive state.json from events.jsonl.

Design
------
- The ONLY source of truth is ``events.jsonl``. ``state.json`` is a derived
  artifact reconstructed from the event log.
- ``rebuild_state()`` reads all events and produces the current state dict.
- Incremental update is NOT implemented — always rebuild from all events.
  Events are small enough that this is negligible (<< 1s for thousands).

Thread safety
-------------
- Called from the main loop (not heartbeat). No lock needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def rebuild_state(events_path: Path) -> dict[str, Any]:
    """Rebuild state.json dict by replaying all events.

    The returned dict follows ``schemas/state.schema.json``.
    """
    if not events_path.exists():
        msg = f"events.jsonl not found: {events_path}"
        raise FileNotFoundError(msg)

    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "",
        "status": "unknown",
        "created_at": "",
        "updated_at": "",
        "config": {},
        "hardware": {},
        "phases": {},
        "results": {},
        "errors": [],
    }

    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        _apply_event(state, event)

    return state


def _apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    """Update ``state`` in-place based on one event."""

    event_name = event.get("event", "")
    data = event.get("data", {})
    ts = event.get("ts", "")

    # Always set run_id from first init event
    if event_name == "init":
        state["run_id"] = event.get("run_id", "")
        state["created_at"] = ts
        state["updated_at"] = ts
        state["config"] = {
            "backend_requested": data.get("backend", ""),
            "backend_resolved": None,
            "max_iter": data.get("max_iter", 200),
            "n_shots": data.get("n_shots", 1024),
            "random_state": data.get("random_state", 42),
        }

    elif event_name == "preflight_ok":
        state["status"] = "running"
        state["updated_at"] = ts
        state["config"]["backend_resolved"] = data.get("backend", "")

    elif event_name == "preflight_failed":
        state["status"] = "failed"
        state["updated_at"] = ts
        err = {"ts": ts, "where": "preflight",
               "what": data.get("error", "unknown")}
        # Deduplicate based on where and what, ignoring ts
        if not any(e.get("where") == err["where"] and e.get("what") == err["what"] for e in state["errors"]):
            state["errors"].append(err)

    elif event_name == "iter_start":
        state["updated_at"] = ts
        iter_idx = data.get("iteration", 0)
        state["phases"][f"iter_{iter_idx}"] = {
            "status": "running", "started_at": ts,
        }

    elif event_name == "iter_complete":
        iter_idx = data.get("iteration", 0)
        phase = state["phases"].get(f"iter_{iter_idx}", {})
        phase["status"] = "completed"
        phase["completed_at"] = ts
        if "cost" in data:
            phase["cost"] = data["cost"]
        if "accuracy" in data:
            phase["accuracy"] = data["accuracy"]
        state["phases"][f"iter_{iter_idx}"] = phase
        state["updated_at"] = ts

    elif event_name == "iter_error":
        iter_idx = data.get("iteration", 0)
        phase = state["phases"].get(f"iter_{iter_idx}", {})
        phase["status"] = "failed"
        phase["failed_at"] = ts
        state["phases"][f"iter_{iter_idx}"] = phase
        err = {"ts": ts, "where": f"iter_{iter_idx}",
               "what": data.get("error", "unknown")}
        # Deduplicate based on where and what, ignoring ts
        if not any(e.get("where") == err["where"] and e.get("what") == err["what"] for e in state["errors"]):
            state["errors"].append(err)
        state["updated_at"] = ts

    elif event_name == "shutdown":
        state["status"] = data.get("reason", "stopped")
        state["updated_at"] = ts

    elif event_name == "complete":
        state["status"] = "completed"
        state["updated_at"] = ts
        if "results" in data:
            state["results"] = data["results"]

    elif event_name == "error":
        state["status"] = "failed"
        state["updated_at"] = ts
        err = {"ts": ts, "where": data.get("where", "unknown"),
               "what": data.get("what", "unknown")}
        # Deduplicate based on where and what, ignoring ts
        if not any(e.get("where") == err["where"] and e.get("what") == err["what"] for e in state["errors"]):
            state["errors"].append(err)