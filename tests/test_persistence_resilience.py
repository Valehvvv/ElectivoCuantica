"""Tests for resilience module (signal, watchdog, heartbeat, shutdown)."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from persistence.resilience import (
    SelfPipeSignal,
    Watchdog,
    Heartbeat,
    GracefulShutdown,
    get_watchdog_timeout,
    BACKEND_TIMEOUTS,
)
from persistence.logger import JsonlEventLogger


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class TestWatchdog:
    def test_not_expired_immediately(self) -> None:
        w = Watchdog(60.0)
        assert not w.expired
        assert w.remaining > 59.0

    def test_expired_after_timeout(self) -> None:
        w = Watchdog(0.01)
        time.sleep(0.02)
        assert w.expired

    def test_tick_raises_on_expired(self) -> None:
        w = Watchdog(0.01)
        time.sleep(0.02)
        with pytest.raises(TimeoutError):
            w.tick()

    def test_reset(self) -> None:
        w = Watchdog(0.01)
        time.sleep(0.02)
        assert w.expired
        w.reset()
        assert not w.expired

    def test_elapsed_increases(self) -> None:
        w = Watchdog(10.0)
        t0 = w.elapsed
        time.sleep(0.01)
        assert w.elapsed > t0


# ---------------------------------------------------------------------------
# Backend timeouts
# ---------------------------------------------------------------------------

class TestBackendTimeouts:
    def test_statevector_timeout(self) -> None:
        assert get_watchdog_timeout("statevector") == 300.0

    def test_spinq_timeout(self) -> None:
        assert get_watchdog_timeout("spinq_nmr") == 1800.0

    def test_unknown_default(self) -> None:
        assert get_watchdog_timeout("unknown") == 3600.0

    def test_unknown_custom_default(self) -> None:
        assert get_watchdog_timeout("unknown", default=100.0) == 100.0


# ---------------------------------------------------------------------------
# Self-pipe signal handler
# ---------------------------------------------------------------------------

class TestSelfPipeSignal:
    def test_enter_exit(self) -> None:
        with SelfPipeSignal(signal.SIGINT) as sp:
            assert sp.fileno >= 0
        # exited cleanly

    def test_fileno_before_enter_raises(self) -> None:
        sp = SelfPipeSignal(signal.SIGINT)
        with pytest.raises(RuntimeError):
            _ = sp.fileno

    def test_wait_timeout_false(self) -> None:
        with SelfPipeSignal(signal.SIGINT) as sp:
            result = sp.wait(timeout=0.01)
            assert not result  # no signal sent

    def test_signal_triggers_wait(self) -> None:
        with SelfPipeSignal(signal.SIGINT) as sp:
            os.kill(os.getpid(), signal.SIGINT)
            result = sp.wait(timeout=1.0)
            assert result

    def test_drain(self) -> None:
        with SelfPipeSignal(signal.SIGINT) as sp:
            os.kill(os.getpid(), signal.SIGINT)
            sp.drain()
            # After drain, wait should timeout
            result = sp.wait(timeout=0.01)
            assert not result


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_start_stop(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        logger = JsonlEventLogger(p)
        hb = Heartbeat(logger, interval_sec=0.05)
        hb.start()
        time.sleep(0.12)
        hb.stop()
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        # At least 2 heartbeats (0.12 / 0.05 = ~2)
        heartbeat_events = [ln for ln in lines if '"heartbeat"' in ln]
        assert len(heartbeat_events) >= 1
        assert not hb.shutdown_requested

    def test_shutdown_flag(self) -> None:
        logger = JsonlEventLogger("/tmp/test_hb_shutdown.jsonl")
        hb = Heartbeat(logger, interval_sec=0.1)
        assert not hb.shutdown_requested
        hb.stop()  # no-op if not started
        assert hb.shutdown_requested


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    def test_initial_state(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        logger = JsonlEventLogger(p)
        gs = GracefulShutdown(logger, tmp_path, "statevector")
        assert not gs.should_stop

    def test_request(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        logger = JsonlEventLogger(p)
        gs = GracefulShutdown(logger, tmp_path, "statevector", ansatz="hea")
        gs.request("interrupted", iteration=5)
        assert gs.should_stop
        assert gs.state.reason == "interrupted"
        assert gs.state.iteration == 5

    def test_double_request_idempotent(self, tmp_path: Path) -> None:
        logger = JsonlEventLogger(tmp_path / "events.jsonl")
        gs = GracefulShutdown(logger, tmp_path, "statevector")
        gs.request("interrupted")
        gs.request("timeout")  # second should be ignored
        assert gs.state.reason == "interrupted"

    def test_logs_shutdown_event(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        logger = JsonlEventLogger(p)
        gs = GracefulShutdown(logger, tmp_path, "spinq_nmr")
        gs.request("timeout", iteration=42)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        shutdown_events = [ln for ln in lines if '"shutdown"' in ln]
        assert len(shutdown_events) == 1
        import json
        ev = json.loads(shutdown_events[0])
        assert ev["data"]["reason"] == "timeout"
        assert ev["data"]["iteration"] == 42

    def test_flush_on_request(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        logger = JsonlEventLogger(p, mode="batched")
        gs = GracefulShutdown(logger, tmp_path, "statevector")
        gs.request("interrupted")
        # The event should be flushed (not batched)
        assert p.stat().st_size > 0