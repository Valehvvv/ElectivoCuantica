"""Durable iteration loop — wraps optimizer with event logging + persistence.

Design
------
- Uses scipy.optimize.minimize with ``method="COBYLA"`` (Phase 2 default).
- On every iteration callback:
  1. Log ``iter_start`` / ``iter_complete`` events via JsonlEventLogger
  2. Save theta array to .npz
  3. Append cost to results.csv
  4. Rebuild state.json from events (snapshot)
- On exception / keyboard interrupt:
  1. Log ``error`` event with traceback
  2. Save current theta
  3. Rebuild final state.json
  4. Re-raise

Thread safety
-------------
- The iteration loop runs in the main thread.
- The heartbeat thread (P3.T3) will NOT touch these files.
- No lock needed for the CSV/theta/snapshot writes.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from persistence.csv_writer import append_results_row
from persistence.logger import FsyncMode, JsonlEventLogger
from persistence.snapshot import rebuild_state
from persistence.theta import save_theta

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]
def _make_tqdm(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable
Callback = Callable[[np.ndarray], None]
class DurableIteration:
    """Wrap an optimizer with event logging and file persistence.

    Parameters
    ----------
    run_dir
        Directory for this run's artifacts (events.jsonl, state.json,
        theta_*.npz, results.csv).
    ansatz
        Name of the ansatz (e.g. ``"hea"``, ``"ansatz_base"``).
    backend
        Backend identifier (e.g. ``"statevector"``, ``"spinq_nmr"``).
    cost_fn
        Function ``cost_fn(theta: np.ndarray) -> float`` that evaluates the
        VQC circuit and returns a scalar loss.
    theta0
        Initial parameter vector.
    n_params
        Expected number of parameters (for CSV and logging).
    logger
        Pre-configured JsonlEventLogger for this run.
    max_iter
        Maximum number of iterations (default 200). Passed to the optimizer
        as ``options["maxiter"]``.
    tol
        Tolerance for optimizer convergence (default 1e-4).
    results_csv_path
        Path to results.csv. Default: ``run_dir / "results.csv"``.
    progress_bar
        Show tqdm progress bar if available (default True).
    """

    def __init__(
        self,
        run_dir: Path,
        ansatz: str,
        backend: str,
        cost_fn: Callable[[np.ndarray], float],
        theta0: np.ndarray,
        n_params: int,
        logger: JsonlEventLogger,
        max_iter: int = 200,
        tol: float = 1e-4,
        results_csv_path: Path | None = None,
        progress_bar: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.ansatz = ansatz
        self.backend = backend
        self.cost_fn = cost_fn
        self.theta0 = np.asarray(theta0, dtype=np.float64)
        self.n_params = n_params
        self.logger = logger
        self.max_iter = max_iter
        self.tol = tol
        self.results_csv_path = results_csv_path or run_dir / "results.csv"
        self.progress_bar = progress_bar and tqdm is not None

        # Tracking
        self.current_iteration = 0
        self.current_theta: np.ndarray | None = None
        self.best_theta: np.ndarray | None = None
        self.best_cost: float = float("inf")

    def run(self) -> dict[str, Any]:
        """Run the durable optimization loop.

        Returns
        -------
        dict with keys:
            - success (bool)
            - final_theta (np.ndarray)
            - final_cost (float)
            - n_iter (int)
            - message (str)
        """
        events_path = self.run_dir / "events.jsonl"

        # Wrap cost_fn for logging
        def _logged_cost(theta: np.ndarray) -> float:
            cost = self.cost_fn(theta)
            self.current_theta = theta.copy()
            return cost

        # Callback called by scipy after each iteration
        def _callback(xk: np.ndarray) -> None:
            self.current_iteration += 1
            cost = self.cost_fn(xk)
            self.current_theta = xk.copy()

            # Log iter
            self.logger.log({
                "ts": self._now_iso(),
                "level": "info",
                "run_id": events_path.parent.name,
                "event": "iter",
                "iter": self.current_iteration,
                "data": {
                    "iteration": self.current_iteration,
                    "cost": float(cost),
                    "params": xk.tolist(),
                },
            })

            # Save theta
            save_theta(self.run_dir, self.ansatz, self.current_iteration, xk, cost)

            # Append to results.csv
            append_results_row(
                self.results_csv_path,
                iteration=self.current_iteration,
                cost=float(cost),
                params=xk.tolist(),
                backend=self.backend,
            )

            # Track best
            if cost < self.best_cost:
                self.best_cost = float(cost)
                self.best_theta = xk.copy()

            # Rebuild state.json from events
            try:
                state = rebuild_state(events_path)
                state_path = self.run_dir / "state.json"
                import json, os
                with state_path.open("w") as fh:
                    json.dump(state, fh, indent=2, default=str)
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                pass  # snapshot is best-effort

        if self.progress_bar:
            def _progress_callback(xk: np.ndarray) -> None:
                _callback(xk)
        else:
            _progress_callback = _callback  # type: ignore[assignment]

        # Log iter for iter 0
        self.logger.log({
            "ts": self._now_iso(),
            "level": "info",
            "run_id": events_path.parent.name,
            "event": "iter",
            "iter": 0,
            "data": {"iteration": 0, "theta0": self.theta0.tolist()},
        })

        try:
            result = minimize(
                _logged_cost,
                self.theta0,
                method="COBYLA",
                callback=_callback,
                options={"maxiter": self.max_iter, "tol": self.tol, "disp": False},
            )
        except KeyboardInterrupt:
            # Graceful shutdown on Ctrl+C
            self.logger.log({
                "ts": self._now_iso(),
                "level": "warning",
                "run_id": events_path.parent.name,
                "event": "shutdown",
                "data": {"reason": "interrupted", "iterator": self.current_iteration},
            })
            if self.current_theta is not None:
                save_theta(self.run_dir, self.ansatz, self.current_iteration, self.current_theta)
            try:
                state = rebuild_state(events_path)
                state_path = self.run_dir / "state.json"
                import json, os
                with state_path.open("w") as fh:
                    json.dump(state, fh, indent=2, default=str)
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                pass
            self.logger.flush()
            return {
                "success": False,
                "final_theta": self.current_theta if self.current_theta is not None else self.theta0,
                "final_cost": self.best_cost if self.best_cost < float("inf") else float("nan"),
                "n_iter": self.current_iteration,
                "message": "KeyboardInterrupt",
            }
        except Exception:
            self.logger.log({
                "ts": self._now_iso(),
                "level": "error",
                "run_id": events_path.parent.name,
                "event": "error",
                "data": {
                    "where": "iteration_loop",
                    "what": traceback.format_exc(),
                },
            })
            if self.current_theta is not None:
                save_theta(self.run_dir, self.ansatz, self.current_iteration, self.current_theta)
            self.logger.flush()
            raise

        # Log completion
        self.logger.log({
            "ts": self._now_iso(),
            "level": "info",
            "run_id": events_path.parent.name,
            "event": "complete",
            "data": {
                "success": bool(result.success),
                "final_cost": float(result.fun),
                "n_iter": self.current_iteration,
                "message": result.message if result.message else "",
            },
        })

        # Final snapshot
        try:
            state = rebuild_state(events_path)
            state_path = self.run_dir / "state.json"
            import json, os
            with state_path.open("w") as fh:
                json.dump(state, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass

        self.logger.flush()

        return {
            "success": bool(result.success),
            "final_theta": result.x,
            "final_cost": float(result.fun),
            "n_iter": self.current_iteration,
            "message": str(result.message),
        }

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime
        return datetime.now().astimezone().isoformat(timespec="milliseconds")