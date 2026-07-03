"""General-purpose utility functions shared across the project."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def timer(name: str = "") -> Iterator[None]:
    """Context manager that prints elapsed time in seconds.

    Usage::

        with timer("training"):
            train_model(...)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    label = f" [{name}]" if name else ""
    print(f"Elapsed{label}: {elapsed:.4f} s")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clamp_probabilities(p: float, eps: float = 1e-10) -> float:
    """Clamp a probability value to ``[eps, 1 - eps]``."""
    return max(eps, min(1.0 - eps, p))
