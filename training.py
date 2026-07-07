"""Training utilities for the VQC models.

Provides the core training loop, BCE loss, and result serialisation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from config import BCE_EPS, DEFAULT_MAX_ITER, INITIAL_THETA_RANGE
from models import VQC

logger = logging.getLogger("vqc")


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary cross-entropy loss with epsilon clipping.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
    """
    y_pred = np.clip(y_pred, BCE_EPS, 1.0 - BCE_EPS)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def train_model(
    vqc: VQC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    initial_theta: np.ndarray | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, Any]:
    """Train a single VQC model and return a result dict.

    Parameters
    ----------
    vqc : VQC
        Configured (but not fitted) classifier.
    X_train : np.ndarray of shape (n, 2)
    y_train : np.ndarray of shape (n,)
    initial_theta : np.ndarray or None
    max_iter : int

    Returns
    -------
    dict with keys: ``theta``, ``loss_history``, ``training_time_sec``,
    ``optimizer_result``.
    """
    start = time.perf_counter()
    vqc.fit(X_train, y_train, initial_theta=initial_theta, max_iter=max_iter)
    elapsed = time.perf_counter() - start

    return {
        "theta": vqc.theta_,
        "loss_history": vqc.history_,
        "training_time_sec": elapsed,
        "optimizer_result": None,  # filled externally if needed
    }


def train_all_ansatze(
    X_train: np.ndarray,
    y_train: np.ndarray,
    backend_mode: str = "statevector",
    backend: Any = None,
    optimizer: str = "COBYLA",
    max_iter: int = DEFAULT_MAX_ITER,
    n_shots: int = 1024,
    registry: dict[str, tuple[Any, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train Base, Reducido, and HEA on the same data with the same backend.

    Parameters
    ----------
    X_train, y_train : arrays
        Training data.
    backend_mode : str
    backend : optional Qiskit backend.
    optimizer : str
    max_iter : int
    n_shots : int
    registry : dict[name -> (builder_fn, n_params)], optional
        Subset of ansätze to train. Defaults to ``models.ansatz_registry()``
        (all three) when ``None``.

    Returns
    -------
    dict mapping ansatz name → training result dict.
    """
    if registry is None:
        from models import ansatz_registry

        registry = ansatz_registry()
    results: dict[str, dict[str, Any]] = {}

    for name, (builder, n_params) in registry.items():
        logger.info("── Training %s ansatz ──", name)

        vqc = VQC(
            ansatz_fn=builder,
            n_params=n_params,
            backend_mode=backend_mode,
            backend=backend,
            optimizer=optimizer,
            n_shots=n_shots,
        )

        result = train_model(vqc, X_train, y_train, max_iter=max_iter)
        result["vqc"] = vqc
        results[name] = result

        logger.info(
            "%s: final_loss=%.6f training_time=%.2fs",
            name,
            result["loss_history"][-1],
            result["training_time_sec"],
        )

    return results
