"""Preprocessing pipeline for the Iris → 2-qubit VQC experiment.

Function contract
-----------------
Each function is independent and composable so that the same pipeline can
be re-run with different configurations (e.g. different class pairs).

Pipeline order
--------------
1. ``load_dataset``
2. ``filter_classes``
3. ``split_dataset``
4. ``apply_lda``   (produces 2-D representation: LDA1 + Petal Width)
5. ``normalize_to_pi``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset() -> pd.DataFrame:
    """Load the Iris dataset as a pandas DataFrame with target names.

    Returns
    -------
    pd.DataFrame
        Iris frame with feature columns, ``"target"`` (integer label),
        and ``"target_name"`` (string label).
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target_name"] = df["target"].apply(lambda t: iris.target_names[t])
    return df


def filter_classes(
    df: pd.DataFrame,
    class_pair: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Keep only the rows belonging to ``class_pair`` and re-map targets.

    Targets are mapped to 0 and 1.

    Parameters
    ----------
    df : pd.DataFrame
        Iris frame from ``load_dataset``.
    class_pair : tuple[int, int]
        Original integer labels to keep, e.g. ``(0, 1)`` for Setosa vs
        Versicolor.

    Returns
    -------
    pd.DataFrame
        Filtered copy with only the selected classes.
    """
    mask = df["target"].isin(class_pair)
    out = df.loc[mask].copy()
    out["target"] = out["target"].map({class_pair[0]: 0, class_pair[1]: 1})
    iris_names = load_iris().target_names
    name_map = {0: str(iris_names[class_pair[0]]), 1: str(iris_names[class_pair[1]])}
    out["target_name"] = out["target"].map(name_map)
    return out


def split_dataset(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split on selected feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered Iris frame.
    feature_names : list[str], optional
        Column names for features. Defaults to the four standard Iris
        features.
    test_size : float
        Fraction reserved for test.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature matrices and label vectors.
    """
    if feature_names is None:
        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

    X = df[list(feature_names)].values.astype(float)
    y = df["target"].values.astype(int)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def apply_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Create a 2-D representation: LDA component 1 + Petal Width.

    The LDA is fit **only** on training data to prevent data leakage.  The
    second dimension is the original "petal width (cm)" column (index 3),
    identified during exploratory analysis as the best single feature for
    Versicolor/Virginica separation.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, 4)
        Training features (all 4 original Iris columns).
    y_train : np.ndarray of shape (n_train,)
        Training labels.
    X_test : np.ndarray of shape (n_test, 4)
        Test features.

    Returns
    -------
    X_train_2d : np.ndarray of shape (n_train, 2)
    X_test_2d : np.ndarray of shape (n_test, 2)
    lda : LinearDiscriminantAnalysis
        Fitted LDA object (useful for later inspection).
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda1 = lda.fit_transform(X_train_std, y_train).ravel()
    X_test_lda1 = lda.transform(X_test_std).ravel()

    petal_width_idx = 3  # "petal width (cm)"
    X_train_2d = np.column_stack([X_train_lda1, X_train[:, petal_width_idx]])
    X_test_2d = np.column_stack([X_test_lda1, X_test[:, petal_width_idx]])

    return X_train_2d, X_test_2d, lda


def normalize_to_pi(
    X_train_2d: np.ndarray,
    X_test_2d: np.ndarray,
    angle_range: tuple[float, float] = (0.0, float(np.pi)),
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale 2-D features into ``[angle_range[0], angle_range[1]]``.

    The scaler is fit on the training set only and then applied to test.

    Parameters
    ----------
    X_train_2d : np.ndarray of shape (n_train, 2)
    X_test_2d : np.ndarray of shape (n_test, 2)
    angle_range : tuple[float, float]
        Target range for angular encoding, default ``(0, pi)``.

    Returns
    -------
    X_train_angle : np.ndarray
    X_test_angle : np.ndarray
    scaler : MinMaxScaler
    """
    scaler = MinMaxScaler(feature_range=angle_range)
    X_train_angle = scaler.fit_transform(X_train_2d)
    X_test_angle = scaler.transform(X_test_2d)
    return X_train_angle, X_test_angle, scaler
