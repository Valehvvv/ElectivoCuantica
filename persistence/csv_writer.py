"""CSV append writers for results and predictions.

Design
------
- Appending a single row at a time (one per iteration).
- File is created with headers on first write if not present.
- Each write is fsynced for durability.

Files:
  - ``results.csv`` — one row per iteration with cost, params, metadata
  - ``predictions.csv`` — one row per sample with predicted label, probability
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


def append_results_row(
    path: Path,
    iteration: int,
    cost: float,
    params: list[float],
    accuracy: float | None = None,
    backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one row to results.csv. Creates file with headers if missing."""
    fieldnames = [
        "iteration", "cost", "accuracy", "backend",
        "params", "timestamp",
    ]
    row: dict[str, Any] = {
        "iteration": iteration,
        "cost": cost,
        "accuracy": accuracy or "",
        "backend": backend or "",
        "params": ",".join(f"{p:.8f}" for p in params),
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        row.update(extra)
    _append_csv(path, fieldnames, row)


def append_predictions_row(
    path: Path,
    iteration: int,
    sample_idx: int,
    label_true: int,
    label_pred: int,
    prob_class1: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one prediction row. Creates file with headers if missing."""
    fieldnames = [
        "iteration", "sample_idx", "label_true", "label_pred",
        "prob_class1", "timestamp",
    ]
    row: dict[str, Any] = {
        "iteration": iteration,
        "sample_idx": sample_idx,
        "label_true": label_true,
        "label_pred": label_pred,
        "prob_class1": f"{prob_class1:.6f}",
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        row.update(extra)
    _append_csv(path, fieldnames, row)


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    """Low-level CSV append with header auto-creation."""
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Include all keys from row in fieldnames to support extra fields
    all_fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())