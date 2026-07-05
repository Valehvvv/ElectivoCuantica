"""Tests for JsonlEventLogger."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from persistence.logger import JsonlEventLogger, BatchConfig


SAMPLE_EVENT = {
    "ts": "2026-07-06T09:15:00.000-03:00",
    "level": "info",
    "run_id": "20260706_091500",
    "event": "init",
    "data": {"backend": "statevector"},
}
@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "events.jsonl"
class TestInit:
    def test_default_mode_is_every_write(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        assert log.mode == "every_write"

    def test_batched_mode(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path, mode="batched")
        assert log.mode == "batched"

    def test_path_property(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        assert log.path == log_path
class TestLog:
    def test_appends_event(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        log.log(SAMPLE_EVENT)
        lines = [ln for ln in log_path.read_text().split("\n") if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "init"

    def test_appends_multiple(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        for i in range(3):
            ev = dict(SAMPLE_EVENT)
            ev["data"] = {"i": i}
            log.log(ev)
        lines = [ln for ln in log_path.read_text().split("\n") if ln.strip()]
        assert len(lines) == 3

    def test_rejects_invalid_event(self, log_path: Path) -> None:
        from schemas.validators import ValidationError
        log = JsonlEventLogger(log_path)
        bad = dict(SAMPLE_EVENT)
        bad["level"] = "fatal"  # invalid enum
        with pytest.raises(ValidationError):
            log.log(bad)
        # File should be empty (nothing was written)
        # File may not exist if no valid writes occurred
        if log_path.exists():
            assert log_path.read_text() == ""
class TestFlush:
    def test_flush_called_does_not_error(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        log.log(SAMPLE_EVENT)
        log.flush()  # should not raise

    def test_close_flushes(self, log_path: Path) -> None:
        log = JsonlEventLogger(log_path)
        log.log(SAMPLE_EVENT)
        log.close()
        assert log_path.exists()
class TestBatchedMode:
    def test_fsync_on_max_events(self, log_path: Path, monkeypatch) -> None:
        fsync_calls = []
        monkeypatch.setattr(
            "persistence.logger.JsonlEventLogger._fsync",
            lambda self: fsync_calls.append(1),
        )
        log = JsonlEventLogger(log_path, mode="batched",
                                batch_config=BatchConfig(max_events=2))
        log.log(SAMPLE_EVENT)
        assert len(fsync_calls) == 0  # not yet
        log.log(SAMPLE_EVENT)
        assert len(fsync_calls) == 1  # reached max_events=2

    def test_fsync_on_timeout(self, log_path: Path, monkeypatch) -> None:
        fsync_calls = []
        monkeypatch.setattr(
            "persistence.logger.JsonlEventLogger._fsync",
            lambda self: fsync_calls.append(1),
        )
        log = JsonlEventLogger(log_path, mode="batched",
                                batch_config=BatchConfig(max_events=100, max_interval_sec=0.01))
        log.log(SAMPLE_EVENT)
        time.sleep(0.02)
        log.log(SAMPLE_EVENT)
        # Interval should have triggered fsync before max_events
        assert len(fsync_calls) == 1