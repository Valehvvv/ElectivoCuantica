"""Theta persistence — save and load theta arrays per ansatz per iteration.

Format
------
- Binary ``.npz`` files (NumPy compressed). One file per iteration.
- File name: ``theta_{ansatz_name}_iter_{iter:04d}.npz``
- Contains: ``theta`` array, ``iteration``, ``ansatz``, ``cost`` (if available).

Design
------
- Theta is saved EVERY iteration, regardless of whether the run completes.
- This enables:
  1. Recovery from interruption (resume from last saved theta)
  2. Post-hoc analysis of the optimization trajectory
  3. Cross-platform comparison (SpinQ vs IBM with same theta init)
- NO intermediate CSV for theta — the npz files ARE the theta log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_theta(
    run_dir: Path,
    ansatz: str,
    iteration: int,
    theta: np.ndarray,
    cost: float | None = None,
) -> Path:
    """Save theta array to ``{run_dir}/theta_{ansatz}_iter_{iteration:04d}.npz``.

    Returns the path of the saved file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"theta_{ansatz}_iter_{iteration:04d}.npz"
    data: dict[str, Any] = {
        "theta": np.asarray(theta, dtype=np.float64),
        "iteration": iteration,
        "ansatz": ansatz,
    }
    if cost is not None:
        data["cost"] = float(cost)
    np.savez_compressed(path, **data)
    return path


def load_theta(path: Path) -> dict[str, Any]:
    """Load a saved theta npz file.

    Returns dict with keys: ``theta`` (np.ndarray), ``iteration`` (int),
    ``ansatz`` (str), optionally ``cost`` (float).
    """
    if not path.exists():
        msg = f"theta file not found: {path}"
        raise FileNotFoundError(msg)
    data = np.load(path)
    result: dict[str, Any] = {
        "theta": data["theta"],
        "iteration": int(data["iteration"]),
        "ansatz": str(data["ansatz"]),
    }
    if "cost" in data:
        result["cost"] = float(data["cost"])
    return result


def latest_theta(run_dir: Path, ansatz: str) -> dict[str, Any] | None:
    """Find the latest saved theta for a given ansatz.

    Returns None if no theta files found for that ansatz.
    """
    pattern = f"theta_{ansatz}_iter_*.npz"
    files = sorted(run_dir.glob(pattern))
    if not files:
        return None
    return load_theta(files[-1])