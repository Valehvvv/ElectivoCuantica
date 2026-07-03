"""Data loading and I/O utilities for the VQC project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_angles_from_csv(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed angle data from CSV files.

    Parameters
    ----------
    train_path : Path
        Path to the training CSV (columns: x0, x1, target).
    test_path : Path
        Path to the test CSV (columns: x0, x1, target).

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature matrices of shape (n_samples, 2) and label vectors.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[["x0", "x1"]].values.astype(float)
    y_train = train_df["target"].values.astype(int)
    X_test = test_df[["x0", "x1"]].values.astype(float)
    y_test = test_df["target"].values.astype(int)

    return X_train, X_test, y_train, y_test


def save_angles_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    filepath: str | Path,
) -> None:
    """Save normalized angle features and labels to CSV.

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)
        Feature matrix in [0, pi] range.
    y : np.ndarray of shape (n,)
        Binary labels.
    filepath : Path
        Destination CSV path.
    """
    df = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "target": y})
    df.to_csv(filepath, index=False)
