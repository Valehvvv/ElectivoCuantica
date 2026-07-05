#!/usr/bin/env python3
"""Skeleton runner for instrumented VQC.

Validates env, runs preflight, writes events.jsonl and state.json.
No training in this phase.

Usage:
    uv run python -m runner.skeleton --backend statevector --pilot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from schemas.validators import write_jsonl_line, validate_event
from preflight.preflight import run_preflight


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VQC skeleton runner")
    parser.add_argument("--backend", default="statevector",
                        help="Backend mode (statevector, spinq_nmr, ...)")
    parser.add_argument("--pilot", action="store_true",
                        help="Reduce params for quick smoke test")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results",
                        help="Root results directory")
    return parser.parse_args(argv)


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = make_run_id()
    run_dir = Path(args.results_dir) / args.backend / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    state_path = run_dir / "state.json"

    # Init event
    init_event = {
        "ts": _now_iso(), "level": "info", "run_id": run_id,
        "event": "init",
        "data": {"backend": args.backend, "pilot": args.pilot},
    }
    write_jsonl_line(events_path, init_event)

    # Initial state
    state = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "created_at": _now_iso(), "updated_at": _now_iso(),
        "config": {
            "backend_requested": args.backend, "backend_resolved": None,
            "max_iter": 5 if args.pilot else 200,
            "n_shots": 64 if args.pilot else 1024,
            "random_state": 42, "endianness": "little",
        },
        "hardware": {},
        "phases": {}, "results": {}, "errors": [],
    }
    write_state(state_path, state)

    # Preflight
    pre = run_preflight(run_id, events_path, backend=args.backend)
    state["config"]["backend_resolved"] = pre.backend_resolved
    if not pre.ok:
        state["status"] = "failed"
        state["errors"].append({
            "ts": _now_iso(), "where": "preflight", "what": pre.error or "unknown",
        })
        write_state(state_path, state)
        return 1

    # Done
    state["status"] = "completed"
    state["updated_at"] = _now_iso()
    write_state(state_path, state)

    complete_event = {
        "ts": _now_iso(), "level": "info", "run_id": run_id,
        "event": "complete", "data": {"phase": "P1_T7_skeleton"},
    }
    write_jsonl_line(events_path, complete_event)

    print(f"[OK] run complete. Dir: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())