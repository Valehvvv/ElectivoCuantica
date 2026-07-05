"""Tests for skeleton runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from runner.skeleton import main, parse_args


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    return tmp_path / "results"


class TestArgparse:
    def test_pilot_flag(self):
        args = parse_args(["--pilot"])
        assert args.pilot is True

    def test_backend_arg(self):
        args = parse_args(["--backend", "spinq_nmr"])
        assert args.backend == "spinq_nmr"

    def test_default_backend(self):
        args = parse_args([])
        assert args.backend == "statevector"


class TestSkeleton:
    def test_statevector_pilot_succeeds(self, tmp_results: Path):
        rc = main(["--backend", "statevector", "--pilot",
                   "--results-dir", str(tmp_results)])
        assert rc == 0

        # Verify state.json
        dirs = list((tmp_results / "statevector").iterdir())
        assert len(dirs) == 1
        state = json.loads((dirs[0] / "state.json").read_text())
        assert state["status"] == "completed"

    def test_spinq_fails_on_missing_spinqit(self, tmp_results: Path):
        with mock.patch("importlib.util.find_spec", return_value=None):
            rc = main(["--backend", "spinq_nmr",
                       "--results-dir", str(tmp_results)])
        assert rc == 1

        dirs = list((tmp_results / "spinq_nmr").iterdir())
        state = json.loads((dirs[0] / "state.json").read_text())
        assert state["status"] == "failed"