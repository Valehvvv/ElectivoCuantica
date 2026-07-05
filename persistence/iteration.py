"""Durable iteration loop — wraps optimizer with event logging + persistence.

Design
------
Supports two optimizer backends:
  - ``"cobyla"`` — uses ``scipy.optimize.minimize(method="COBYLA")``
  - ``"spsa"`` — uses custom ``optimizers.spsa.minimize_spsa``

On every iteration callback:
  1. Log ``iter_start`` / ``iter_complete`` events via JsonlEventLogger
  2. Save theta array to .npz
  3. Append cost to results.csv
  4. Rebuild state.json from events (snapshot)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from persistence.csv_writer import append_results_row
from persistence.logger import FsyncMode, JsonlEventLogger
from persistence.snapshot import rebuild_state
from persistence.theta import save_theta

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
OptimizerMethod = Literal["cobyla", "spsa"]
def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
class DurableIteration:
    """Wrap an optimizer with event logging and file persistence.

    Parameters
    ----------
    run_dir
        Directory for this run's artifacts.
    ansatz
        Name of the ansatz (e.g. ``"hea"``, ``"ansatz_base"``).
    backend
        Backend identifier.
    cost_fn
        Function ``cost_fn(theta: np.ndarray) -> float``.
    theta0
        Initial parameter vector.
    n_params
        Expected number of parameters.
    logger
        Pre-configured JsonlEventLogger for this run.
    method
        Optimizer backend: ``"cobyla"`` or ``"spsa"``.
    max_iter
        Maximum number of iterations (default 200).
    tol
        Tolerance for convergence (default 1e-4).
    results_csv_path
        Path to results.csv. Default: ``run_dir / "results.csv"``.
    progress_bar
        Show tqdm progress bar if available (default True).
    spsa_kwargs
        Extra kwargs passed to ``minimize_spsa`` (e.g. a, c, alpha, gamma).
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
        method: OptimizerMethod = "cobyla",
        max_iter: int = 200,
        tol: float = 1e-4,
        results_csv_path: Path | None = None,
        progress_bar: bool = True,
        spsa_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.ansatz = ansatz
        self.backend = backend
        self.cost_fn = cost_fn
        self.theta0 = np.asarray(theta0, dtype=np.float64)
        self.n_params = n_params
        self.logger = logger
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.results_csv_path = results_csv_path or run_dir / "results.csv"
        self.progress_bar = progress_bar and tqdm is not None
        self.spsa_kwargs = spsa_kwargs or {}

        # Tracking
        self.current_iteration = 0
        self.current_theta: np.ndarray | None = None
        self.best_theta: np.ndarray | None = None
        self.best_cost: float = float("inf")

    def run(self) -> dict[str, Any]:
        """Run the durable optimization loop.

        Returns dict with keys: success, final_theta, final_cost, n_iter, message.
        """
        if self.method == "cobyla":
            return self._run_cobyla()
        elif self.method == "spsa":
            return self._run_spsa()
        else:
            msg = f"Unknown optimizer method: {self.method}"
            raise ValueError(msg)

    # ── COBYLA ──────────────────────────────────────────────────────

    def _run_cobyla(self) -> dict[str, Any]:
        events_path = self.run_dir / "events.jsonl"

        def _callback(xk: np.ndarray) -> None:
            self._on_iteration(xk)

        self._log_iter_start(0, self.theta0)

        try:
            from scipy.optimize import minimize
            result = minimize(
                self.cost_fn,
                self.theta0,
                method="COBYLA",
                callback=_callback,
                options={"maxiter": self.max_iter, "tol": self.tol, "disp": False},
            )
        except KeyboardInterrupt:
            return self._handle_interrupt()
        except Exception:
            self._handle_error()
            raise

        self._log_complete(result.success, result.fun, result.message)

        # Final snapshot
        self._write_snapshot(events_path)

        self.logger.flush()
        return {
            "success": bool(result.success),
            "final_theta": result.x,
            "final_cost": float(result.fun),
            "n_iter": self.current_iteration,
            "message": str(result.message),
        }

    # ── SPSA ────────────────────────────────────────────────────────

    def _run_spsa(self) -> dict[str, Any]:
        events_path = self.run_dir / "events.jsonl"

        def _callback(xk: np.ndarray) -> None:
            self._on_iteration(xk)

        self._log_iter_start(0, self.theta0)

        try:
            from optimizers.spsa import minimize_spsa
            result = minimize_spsa(
                self.cost_fn,
                self.theta0,
                max_iter=self.max_iter,
                tol=self.tol,
                callback=_callback,
                **self.spsa_kwargs,
            )
        except KeyboardInterrupt:
            return self._handle_interrupt()
        except Exception:
            self._handle_error()
            raise

        self._log_complete(result.success, result.fun, result.message)
        self._write_snapshot(events_path)
        self.logger.flush()
        return {
            "success": result.success,
            "final_theta": result.x,
            "final_cost": result.fun,
            "n_iter": result.n_iter,
            "message": result.message,
        }

    # ── Shared helpers ──────────────────────────────────────────────

    def _on_iteration(self, xk: np.ndarray) -> None:
        self.current_iteration += 1
        cost = float(self.cost_fn(xk))
        self.current_theta = xk.copy()

        self.logger.log({
            "ts": _now_iso(),
            "level": "info",
            "run_id": self.run_dir.name,
            "event": "iter",
            "data": {
                "iteration": self.current_iteration,
                "cost": cost,
                "params": xk.tolist(),
            },
        })

        save_theta(self.run_dir, self.ansatz, self.current_iteration, xk, cost)

        append_results_row(
            self.results_csv_path,
            iteration=self.current_iteration,
            cost=cost,
            params=xk.tolist(),
            backend=self.backend,
        )

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_theta = xk.copy()

        events_path = self.run_dir / "events.jsonl"
        self._write_snapshot(events_path)

    def _log_iter_start(self, iteration: int, theta: np.ndarray) -> None:
        self.logger.log({
            "ts": _now_iso(),
            "level": "info",
            "run_id": self.run_dir.name,
            "event": "iter",
            "data": {"iteration": iteration, "theta0": theta.tolist()},
        })

    def _log_complete(self, success: bool, final_cost: float, message: Any) -> None:
        self.logger.log({
            "ts": _now_iso(),
            "level": "info",
            "run_id": self.run_dir.name,
            "event": "complete",
            "data": {
                "success": success,
                "final_cost": final_cost,
                "n_iter": self.current_iteration,
                "message": str(message) if message else "",
            },
        })

    def _handle_interrupt(self) -> dict[str, Any]:
        self.logger.log({
            "ts": _now_iso(),
            "level": "warning",
            "run_id": self.run_dir.name,
            "event": "shutdown",
            "data": {"reason": "interrupted", "iteration": self.current_iteration},
        })
        if self.current_theta is not None:
            save_theta(self.run_dir, self.ansatz, self.current_iteration, self.current_theta)
        self._write_snapshot(self.run_dir / "events.jsonl")
        self.logger.flush()
        return {
            "success": False,
            "final_theta": self.current_theta if self.current_theta is not None else self.theta0,
            "final_cost": self.best_cost if self.best_cost < float("inf") else float("nan"),
            "n_iter": self.current_iteration,
            "message": "KeyboardInterrupt",
        }

    def _handle_error(self) -> None:
        self.logger.log({
            "ts": _now_iso(),
            "level": "error",
            "run_id": self.run_dir.name,
            "event": "error",
            "data": {"where": "iteration_loop", "what": traceback.format_exc()},
        })
        if self.current_theta is not None:
            save_theta(self.run_dir, self.ansatz, self.current_iteration, self.current_theta)
        self.logger.flush()

    def _write_snapshot(self, events_path: Path) -> None:
        try:
            state = rebuild_state(events_path)
            state_path = self.run_dir / "state.json"
            with state_path.open("w") as fh:
                json.dump(state, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass