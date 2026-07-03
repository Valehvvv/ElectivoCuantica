"""Evaluation metrics for the VQC classification experiment.

Computes a comprehensive set of classification and circuit-structural
metrics for each trained ansatz.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from circuits import ansatz_circuit_info
from models import VQC
from training import binary_cross_entropy


def evaluate_model(
    vqc: VQC,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Compute all classification metrics for a trained VQC.

    Parameters
    ----------
    vqc : VQC (already fitted).
    X_test : np.ndarray of shape (n, 2)
    y_test : np.ndarray of shape (n,)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix,
    inference_time_sec, test_loss.
    """
    t0 = time.perf_counter()
    probs = vqc.predict_proba(X_test)
    preds = vqc.predict(X_test)
    inference_time = time.perf_counter() - t0

    test_loss = binary_cross_entropy(y_test, probs)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "inference_time_sec": inference_time,
        "test_loss": test_loss,
    }


def structural_info(
    vqc: VQC,
    sample_x: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract structural metrics from the ansatz circuit.

    Parameters
    ----------
    vqc : VQC (must be fitted).
    sample_x : np.ndarray of length 2 or None
        Sample features for circuit construction. If ``None``, uses ``[0, 0]``.

    Returns
    -------
    dict from ``ansatz_circuit_info``.
    """
    if sample_x is None:
        sample_x = np.zeros(vqc.n_params)  # fallback zero features

    qc = vqc.ansatz_fn(list(sample_x[:2]), vqc.theta_)
    return ansatz_circuit_info(qc)


def compute_total_time(
    training_time: float,
    inference_time: float,
) -> float:
    """Sum training and inference times."""
    return training_time + inference_time
