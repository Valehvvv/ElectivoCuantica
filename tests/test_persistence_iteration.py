"""Tests for DurableIteration loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from persistence.iteration import DurableIteration
from persistence.logger import JsonlEventLogger
def parabola_cost(theta: np.ndarray) -> float:
    """Simple quadratic for testing: f(x) = sum(x_i^2). Minimum at 0."""
    return float(np.sum(theta ** 2))
class TestDurableIteration:
    @pytest.fixture
    def run_dir(self, tmp_path: Path) -> Path:
        # Create a directory with a valid run_id format (timestamp-based)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        d = tmp_path / timestamp
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def logger(self, run_dir: Path) -> JsonlEventLogger:
        return JsonlEventLogger(run_dir / "events.jsonl")

    def test_optimizes_parabola(self, run_dir: Path) -> None:
        """Minimize f(x)=x^2 starting from theta=[2.0]. Should converge near 0."""
        logger = JsonlEventLogger(run_dir / "events.jsonl")
        loop = DurableIteration(
            run_dir=run_dir,
            ansatz="test",
            backend="statevector",
            cost_fn=parabola_cost,
            theta0=np.array([2.0]),
            n_params=1,
            logger=logger,
            max_iter=50,
            tol=1e-6,
            progress_bar=False,
        )
        result = loop.run()
        assert result["success"]
        assert abs(result["final_theta"][0]) < 0.1  # should converge near 0
        assert result["final_cost"] < 0.1
        assert result["n_iter"] > 0

    def test_events_logged(self, run_dir: Path) -> None:
        logger = JsonlEventLogger(run_dir / "events.jsonl")
        loop = DurableIteration(
            run_dir=run_dir,
            ansatz="test",
            backend="statevector",
            cost_fn=parabola_cost,
            theta0=np.array([2.0]),
            n_params=1,
            logger=logger,
            max_iter=10,
            tol=1e-6,
            progress_bar=False,
        )
        loop.run()
        events_path = run_dir / "events.jsonl"
        lines = [ln for ln in events_path.read_text().splitlines() if ln.strip()]
        events = [json.loads(ln) for ln in lines]
        # Should have: iter N times, complete
        event_names = [e["event"] for e in events]
        assert "iter" in event_names
        # at least one iter
        iters = [e for e in events if e["event"] == "iter"]
        assert len(iters) >= 1
        # final complete
        assert "complete" in event_names

    def test_theta_files_created(self, run_dir: Path) -> None:
        logger = JsonlEventLogger(run_dir / "events.jsonl")
        loop = DurableIteration(
            run_dir=run_dir,
            ansatz="test_theta",
            backend="statevector",
            cost_fn=parabola_cost,
            theta0=np.array([2.0]),
            n_params=1,
            logger=logger,
            max_iter=5,
            tol=1e-10,
            progress_bar=False,
        )
        loop.run()
        npz_files = list(run_dir.glob("theta_test_theta_iter_*.npz"))
        assert len(npz_files) >= 1

    def test_results_csv_created(self, run_dir: Path) -> None:
        logger = JsonlEventLogger(run_dir / "events.jsonl")
        loop = DurableIteration(
            run_dir=run_dir,
            ansatz="test",
            backend="statevector",
            cost_fn=parabola_cost,
            theta0=np.array([2.0]),
            n_params=1,
            logger=logger,
            max_iter=5,
            tol=1e-10,
            progress_bar=False,
        )
        loop.run()
        csv_path = run_dir / "results.csv"
        assert csv_path.exists()
        lines = [ln for ln in csv_path.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2  # header + at least 1 row

    def test_state_json_snapshot(self, run_dir: Path) -> None:
        logger = JsonlEventLogger(run_dir / "events.jsonl")
        loop = DurableIteration(
            run_dir=run_dir,
            ansatz="test",
            backend="statevector",
            cost_fn=parabola_cost,
            theta0=np.array([2.0]),
            n_params=1,
            logger=logger,
            max_iter=5,
            tol=1e-10,
            progress_bar=False,
        )
        loop.run()
        state_path = run_dir / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["status"] in ("completed", "running")
        assert "config" in state
        assert "phases" in state