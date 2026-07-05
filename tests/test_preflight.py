"""Tests for preflight module. Mocks all SDKs; no real imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from preflight.preflight import (
    PreflightResult,
    _BACKEND_CHECKS,
    _check_spinq,
    _check_ibm,
    _check_aer,
    _check_statevector,
    run_preflight,
    _emit_event,
    _now_iso,
    _resolve_backend,
)
from schemas.validators import iter_jsonl


# --- Helpers ------------------------------------------------------------------

def _make_fake_module(name: str, version: str = "1.2.3") -> mock.MagicMock:
    """Create a fake module that exposes ``__version__``."""
    mod = mock.MagicMock()
    mod.__version__ = version
    return mod


@pytest.fixture
def events_path(tmp_path: Path) -> Path:
    return tmp_path / "events.jsonl"


# --- Tests for individual backend checks --------------------------------------

class TestCheckStatevector:
    def test_ok_when_qiskit_present(self) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            ok, info = _check_statevector()
        assert ok is True
        assert info == "1.0.0"

    def test_fail_when_qiskit_missing(self) -> None:
        # Force ImportError by patching importlib
        with mock.patch.dict(sys.modules, {"qiskit": None}):
            ok, info = _check_statevector()
        assert ok is False
        assert "qiskit" in info.lower()


class TestCheckAer:
    def test_ok_when_aer_present(self) -> None:
        with mock.patch.dict(sys.modules, {"qiskit_aer": _make_fake_module("qiskit_aer", "0.13.0")}):
            ok, info = _check_aer()
        assert ok is True
        assert info == "0.13.0"

    def test_fail_when_aer_missing(self) -> None:
        with mock.patch.dict(sys.modules, {"qiskit_aer": None}):
            ok, info = _check_aer()
        assert ok is False
        assert "qiskit_aer" in info.lower()


class TestCheckSpinq:
    def test_fail_when_spinqit_missing(self) -> None:
        # When find_spec returns None
        with mock.patch("importlib.util.find_spec", return_value=None):
            ok, info = _check_spinq()
        assert ok is False
        assert "spinqit" in info.lower()
        assert "docs/spinq_setup.md" in info

    def test_ok_when_spinqit_importable(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=mock.MagicMock()):
            with mock.patch.dict(sys.modules, {"spinqit": _make_fake_module("spinqit", "0.4.2")}):
                ok, info = _check_spinq()
        assert ok is True
        assert info == "0.4.2"


class TestCheckIbm:
    def test_ok_when_ibm_runtime_present(self) -> None:
        with mock.patch.dict(sys.modules, {
            "qiskit_ibm_runtime": _make_fake_module("qiskit_ibm_runtime", "0.20.0"),
        }):
            ok, info = _check_ibm()
        assert ok is True
        assert info == "0.20.0"

    def test_fail_when_ibm_runtime_missing(self) -> None:
        with mock.patch.dict(sys.modules, {"qiskit_ibm_runtime": None}):
            ok, info = _check_ibm()
        assert ok is False
        assert "qiskit_ibm_runtime" in info.lower()


# --- Tests for run_preflight (integration of checks + event emission) ---------

class TestRunPreflight:
    RUN_ID = "20260706_091500"

    def _read_events(self, path: Path) -> list[dict]:
        return [ev for _, ev in iter_jsonl(path)]

    def test_statevector_ok_emits_start_and_ok(self, events_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            result = run_preflight(self.RUN_ID, events_path, backend="statevector")

        assert result.ok is True
        assert result.backend == "statevector"
        assert result.backend_resolved == "statevector"
        assert result.sdk_version == "1.0.0"
        assert result.error is None

        events = self._read_events(events_path)
        assert [e["event"] for e in events] == ["preflight_start", "preflight_ok"]
        assert events[0]["data"]["backend"] == "statevector"
        assert events[1]["data"]["sdk_version"] == "1.0.0"

    def test_spinq_missing_emits_start_and_failed(self, events_path: Path) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None):
            result = run_preflight(self.RUN_ID, events_path, backend="spinq_nmr")

        assert result.ok is False
        assert result.backend == "spinq_nmr"
        assert result.backend_resolved == "spinq_nmr"  # NO fallback
        assert result.sdk_version is None
        assert "spinqit" in result.error.lower()

        events = self._read_events(events_path)
        assert [e["event"] for e in events] == ["preflight_start", "preflight_failed"]
        assert events[1]["level"] == "error"

    def test_unknown_backend_emits_failed(self, events_path: Path) -> None:
        result = run_preflight(self.RUN_ID, events_path, backend="totally_made_up")

        assert result.ok is False
        assert "unknown" in result.error.lower()
        events = self._read_events(events_path)
        assert events[-1]["event"] == "preflight_failed"
        assert "valid:" in events[-1]["data"]["reason"].lower()

    def test_uses_env_var_when_backend_none(self, events_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            with mock.patch.dict("os.environ", {"BACKEND_MODE": "statevector"}):
                result = run_preflight(self.RUN_ID, events_path)
        assert result.ok is True
        assert result.backend == "statevector"

    def test_default_is_statevector(self, events_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            with mock.patch.dict("os.environ", {}, clear=True):
                result = run_preflight(self.RUN_ID, events_path)
        assert result.ok is True
        assert result.backend == "statevector"

    def test_no_fallback_for_unavailable_backend(self, events_path: Path) -> None:
        """CRITICAL: if spinq_nmr is requested and spinqit is missing,
        the result MUST say ok=False with backend_resolved=spinq_nmr.
        It must NOT silently resolve to statevector."""
        with mock.patch("importlib.util.find_spec", return_value=None):
            with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
                # Even though statevector would be OK, we must NOT use it
                result = run_preflight(self.RUN_ID, events_path, backend="spinq_nmr")

        assert result.ok is False
        assert result.backend_resolved == "spinq_nmr"  # not statevector

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "events.jsonl"
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            result = run_preflight(self.RUN_ID, nested, backend="statevector")
        assert result.ok is True
        assert nested.exists()

    def test_appends_to_existing_events(self, events_path: Path) -> None:
        # Pre-populate with an init event
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            run_preflight(self.RUN_ID, events_path, backend="statevector")
            run_preflight(self.RUN_ID, events_path, backend="statevector")

        events = self._read_events(events_path)
        # 2 preflight_starts + 2 preflight_oks = 4
        assert len(events) == 4
        assert events[0]["event"] == "preflight_start"
        assert events[-1]["event"] == "preflight_ok"

    def test_run_id_present_in_events(self, events_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            run_preflight(self.RUN_ID, events_path, backend="statevector")
        events = self._read_events(events_path)
        for e in events:
            assert e["run_id"] == self.RUN_ID

    def test_timestamp_has_tz_offset(self, events_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"qiskit": _make_fake_module("qiskit", "1.0.0")}):
            run_preflight(self.RUN_ID, events_path, backend="statevector")
        events = self._read_events(events_path)
        for e in events:
            # Must contain +/-HH:MM at the end
            assert e["ts"][-6] in ("+", "-"), f"no TZ offset in {e['ts']}"
            assert e["ts"][-3] == ":", f"TZ offset not HH:MM in {e['ts']}"