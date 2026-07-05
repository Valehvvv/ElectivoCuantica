"""Tests for StateSnapshot (rebuild_state)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from persistence.snapshot import rebuild_state


def _write_events(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


class TestRebuildState:
    def test_raises_on_missing_file(self) -> None:
        p = Path("/tmp/nonexistent_events.jsonl")
        with pytest.raises(FileNotFoundError):
            rebuild_state(p)

    def test_init_sets_config(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {
                "ts": "2026-07-06T09:00:00Z",
                "level": "info",
                "run_id": "test_run_001",
                "event": "init",
                "data": {"backend": "statevector", "pilot": True},
            },
        ])
        state = rebuild_state(p)
        assert state["run_id"] == "test_run_001"
        assert state["config"]["backend_requested"] == "statevector"
        assert state["status"] == "unknown"  # no preflight yet

    def test_preflight_ok_sets_running(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {
                "ts": "2026-07-06T09:00:00Z",
                "level": "info", "run_id": "r1",
                "event": "init",
                "data": {"backend": "statevector"},
            },
            {
                "ts": "2026-07-06T09:00:01Z",
                "level": "info", "run_id": "r1",
                "event": "preflight_ok",
                "data": {"backend": "statevector", "sdk_version": "1.2"},
            },
        ])
        state = rebuild_state(p)
        assert state["status"] == "running"
        assert state["config"]["backend_resolved"] == "statevector"

    def test_preflight_failed(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {
                "ts": "2026-07-06T09:00:00Z",
                "level": "info", "run_id": "r1",
                "event": "init",
                "data": {"backend": "spinq_nmr"},
            },
            {
                "ts": "2026-07-06T09:00:01Z",
                "level": "error", "run_id": "r1",
                "event": "preflight_failed",
                "data": {"error": "spinqit not importable"},
            },
        ])
        state = rebuild_state(p)
        assert state["status"] == "failed"
        assert len(state["errors"]) == 1
        assert "spinqit" in state["errors"][0]["what"]

    def test_iteration_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {"ts": "t0", "level": "info", "run_id": "r1",
             "event": "init", "data": {"backend": "statevector"}},
            {"ts": "t1", "level": "info", "run_id": "r1",
             "event": "preflight_ok",
             "data": {"backend": "statevector", "sdk_version": "1.2"}},
            {"ts": "t2", "level": "info", "run_id": "r1",
             "event": "iter_start", "data": {"iteration": 0}},
            {"ts": "t3", "level": "info", "run_id": "r1",
             "event": "iter_complete", "data": {"iteration": 0, "cost": 0.5}},
        ])
        state = rebuild_state(p)
        assert state["phases"]["iter_0"]["status"] == "completed"
        assert state["phases"]["iter_0"]["cost"] == 0.5
        assert state["status"] == "running"  # not yet complete

    def test_complete_sets_final_status(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {"ts": "t0", "level": "info", "run_id": "r1",
             "event": "init", "data": {"backend": "statevector"}},
            {"ts": "t1", "level": "info", "run_id": "r1",
             "event": "preflight_ok",
             "data": {"backend": "statevector"}},
            {"ts": "t2", "level": "info", "run_id": "r1",
             "event": "complete", "data": {}},
        ])
        state = rebuild_state(p)
        assert state["status"] == "completed"

    def test_error_event_appended(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {"ts": "t0", "level": "info", "run_id": "r1",
             "event": "init", "data": {"backend": "statevector"}},
            {"ts": "t1", "level": "error", "run_id": "r1",
             "event": "error",
             "data": {"where": "optimizer", "what": "SPSA diverged"}},
        ])
        state = rebuild_state(p)
        assert state["status"] == "failed"
        assert any("SPSA diverged" in e["what"] for e in state["errors"])

    def test_shutdown_sets_status(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {"ts": "t0", "level": "info", "run_id": "r1",
             "event": "init", "data": {"backend": "statevector"}},
            {"ts": "t1", "level": "info", "run_id": "r1",
             "event": "shutdown",
             "data": {"reason": "interrupted"}},
        ])
        state = rebuild_state(p)
        assert state["status"] == "interrupted"

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("")
        state = rebuild_state(p)
        assert state["status"] == "unknown"

    def test_duplicate_errors_deduped(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        _write_events(p, [
            {"ts": "t0", "level": "info", "run_id": "r1",
             "event": "init", "data": {"backend": "statevector"}},
            {"ts": "t1", "level": "error", "run_id": "r1",
             "event": "error",
             "data": {"where": "x", "what": "same error"}},
            {"ts": "t2", "level": "error", "run_id": "r1",
             "event": "error",
             "data": {"where": "x", "what": "same error"}},
        ])
        state = rebuild_state(p)
        assert len(state["errors"]) == 1  # deduped