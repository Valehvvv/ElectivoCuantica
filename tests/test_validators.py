"""Tests for schemas/validators.py. Pure mock data, no I/O on hardware."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.validators import (
    ValidationError,
    iter_jsonl,
    validate_event,
    validate_state,
    write_jsonl_line,
)
# --- Sample fixtures (all mocked, no real backend) ----------------------------

def _ts(h: int = 9, m: int = 15, s: int = 0) -> str:
    return f"2026-07-06T{h:02d}:{m:02d}:{s:02d}.000-03:00"
VALID_EVENTS = {
    "init": {
        "ts": _ts(), "level": "info", "run_id": "20260706_091500",
        "event": "init",
        "data": {"backend": "spinq_nmr", "max_iter": 200, "n_shots": 1024},
    },
    "iter": {
        "ts": _ts(9, 15, 10), "level": "info", "run_id": "20260706_091500",
        "event": "iter", "ansatz": "HEA", "iter": 1,
        "data": {"loss": 0.693, "theta_hash": "sha256:abcd"},
    },
    "ansatz_start": {
        "ts": _ts(9, 15, 9), "level": "info", "run_id": "20260706_091500",
        "event": "ansatz_start", "ansatz": "Base",
        "data": {"n_params": 6, "optimizer": "COBYLA"},
    },
    "calibration_done": {
        "ts": _ts(9, 15, 8), "level": "info", "run_id": "20260706_091500",
        "event": "calibration_done",
        "data": {"shots_per_sec": 2.4, "calibration_circuit": "R_y(0) on q0"},
    },
    "shutdown": {
        "ts": _ts(10, 30, 0), "level": "info", "run_id": "20260706_091500",
        "event": "shutdown",
        "data": {"reason": "user_interrupt", "status": "interrupted",
                 "current_phase": "train_hea", "current_ansatz": "HEA",
                 "current_iter": 187},
    },
    "heartbeat": {
        "ts": _ts(9, 16, 0), "level": "debug", "run_id": "20260706_091500",
        "event": "heartbeat",
    },
}

VALID_STATE = {
    "schema_version": 1,
    "run_id": "20260706_091500",
    "status": "running",
    "created_at": _ts(),
    "updated_at": _ts(),
    "config": {
        "backend_requested": "spinq_nmr",
        "backend_resolved": "spinq_nmr",
        "max_iter": 200,
        "n_shots": 1024,
        "random_state": 42,
        "ansatze": ["Base", "Reducido", "HEA"],
        "optimizers": ["COBYLA", "SPSA"],
        "endianness": "little",
    },
    "hardware": {
        "spinqit_version": "0.4.2",
        "shots_per_sec": 2.4,
    },
}
# --- Tests --------------------------------------------------------------------

class TestValidateEvent:
    @pytest.mark.parametrize("name", list(VALID_EVENTS))
    def test_valid_event(self, name: str) -> None:
        ev = VALID_EVENTS[name]
        out = validate_event(ev)
        assert out is ev  # returned for chaining

    def test_missing_ts_raises(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev.pop("ts")
        with pytest.raises(ValidationError, match="'ts' is a required property"):
            validate_event(ev)

    def test_missing_level_raises(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev.pop("level")
        with pytest.raises(ValidationError, match="'level'"):
            validate_event(ev)

    def test_invalid_level_raises(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev["level"] = "fatal"
        with pytest.raises(ValidationError, match="'level'"):
            validate_event(ev)

    def test_invalid_event_raises(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev["event"] = "made_up_event"
        with pytest.raises(ValidationError, match="'event'"):
            validate_event(ev)

    def test_invalid_run_id_raises(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev["run_id"] = "not-a-timestamp-id"
        with pytest.raises(ValidationError, match="'run_id'"):
            validate_event(ev)

    def test_ts_must_have_tz_offset(self) -> None:
        ev = dict(VALID_EVENTS["init"])
        ev["ts"] = "2026-07-06T09:15:00"  # missing TZ offset
        with pytest.raises(ValidationError, match="'ts'"):
            validate_event(ev)

    def test_negative_iter_rejected(self) -> None:
        ev = dict(VALID_EVENTS["iter"])
        ev["iter"] = -1
        with pytest.raises(ValidationError, match="'iter'"):
            validate_event(ev)

    def test_invalid_ansatz_rejected(self) -> None:
        ev = dict(VALID_EVENTS["ansatz_start"])
        ev["ansatz"] = "TotallyMadeUp"
        with pytest.raises(ValidationError, match="'ansatz'"):
            validate_event(ev)
class TestValidateState:
    def test_valid_state(self) -> None:
        out = validate_state(VALID_STATE)
        assert out is VALID_STATE

    def test_missing_status_raises(self) -> None:
        st = dict(VALID_STATE)
        st.pop("status")
        with pytest.raises(ValidationError, match="'status'"):
            validate_state(st)

    def test_invalid_status_raises(self) -> None:
        st = dict(VALID_STATE)
        st["status"] = "aborted"
        with pytest.raises(ValidationError, match="'status'"):
            validate_state(st)

    def test_schema_version_must_be_1(self) -> None:
        st = dict(VALID_STATE)
        st["schema_version"] = 2
        with pytest.raises(ValidationError, match="schema_version"):
            validate_state(st)

    def test_empty_ansatze_rejected(self) -> None:
        st = json.loads(json.dumps(VALID_STATE))
        st["config"]["ansatze"] = []
        with pytest.raises(ValidationError, match="ansatze"):
            validate_state(st)

    def test_invalid_optimizer_rejected(self) -> None:
        st = json.loads(json.dumps(VALID_STATE))
        st["config"]["optimizers"] = ["Adam"]
        with pytest.raises(ValidationError, match="optimizers"):
            validate_state(st)

    def test_shots_must_be_positive(self) -> None:
        st = json.loads(json.dumps(VALID_STATE))
        st["config"]["n_shots"] = 0
        with pytest.raises(ValidationError, match="n_shots"):
            validate_state(st)
class TestIterJsonl:
    def test_iter_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text(
            json.dumps(VALID_EVENTS["init"]) + "\n"
            + "\n"  # blank
            + "   \n"  # whitespace only
            + json.dumps(VALID_EVENTS["iter"]) + "\n"
        )
        out = list(iter_jsonl(p))
        assert len(out) == 2
        assert out[0][0] == 1
        assert out[1][0] == 4  # line 4 (after the blanks)

    def test_iter_reports_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text('{"ts": "2026-07-06T09:15:00.000-03:00", broken\n')
        with pytest.raises(ValidationError, match=r"events\.jsonl:1: invalid JSON"):
            list(iter_jsonl(p))

    def test_iter_reports_invalid_event_with_line(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text(
            json.dumps(VALID_EVENTS["init"]) + "\n"
            + '{"ts": "bad-ts", "level": "info", "run_id": "20260706_091500", "event": "init"}\n'
        )
        with pytest.raises(ValidationError, match=r"events\.jsonl:2: event validation failed at 'ts'"):
            list(iter_jsonl(p))
class TestWriteJsonlLine:
    def test_appends_one_line(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        write_jsonl_line(p, VALID_EVENTS["init"])
        write_jsonl_line(p, VALID_EVENTS["iter"])
        text = p.read_text()
        lines = [ln for ln in text.split("\n") if ln.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "init"
        assert json.loads(lines[1])["event"] == "iter"

    def test_rejects_invalid_before_writing(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        bad = dict(VALID_EVENTS["init"])
        bad["level"] = "fatal"
        with pytest.raises(ValidationError):
            write_jsonl_line(p, bad)
        # File should not exist (never touched)
        assert not p.exists()

    def test_compact_json_no_indent(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        write_jsonl_line(p, VALID_EVENTS["init"])
        text = p.read_text().rstrip("\n")
        # Compact: no spaces after colons or commas
        assert ": " not in text
        assert ", " not in text