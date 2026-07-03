"""Automatic visualisation generation for the VQC experiment.

All plots are saved to ``config.RESULTS_DIR`` by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import RESULTS_DIR


def plot_circuits(
    circuits: dict[str, Any],
    save_path: str | Path | None = None,
) -> None:
    """Draw and save all ansatz circuit diagrams.

    Parameters
    ----------
    circuits : dict[name → QuantumCircuit]
    save_path : Path, optional
    """
    for name, qc in circuits.items():
        fig = qc.draw("mpl", scale=0.7)
        if save_path is None:
            save_path = RESULTS_DIR / f"circuit_{name.lower()}.png"
        else:
            save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved circuit diagram: {save_path}")


def plot_confusion_matrices(
    evaluations: dict[str, dict[str, Any]],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot confusion matrices for each ansatz.

    Parameters
    ----------
    evaluations : dict[name → eval_dict]
        Each eval dict must contain ``"confusion_matrix"``.
    class_names : list[str], optional
    save_path : Path, optional
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    n = len(evaluations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, evaluations.items()):
        cm = np.array(metrics["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"{name} - Confusion Matrix")

    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrices.png"
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrices: {save_path}")


def plot_training_curves(
    training_results: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
) -> None:
    """Plot loss vs. iteration for all trained ansätze.

    Parameters
    ----------
    training_results : dict[name → {"loss_history": [float, ...], ...}]
    save_path : Path, optional
    """
    plt.figure(figsize=(8, 5))
    for name, result in training_results.items():
        plt.plot(result["loss_history"], label=name)

    plt.xlabel("Evaluation")
    plt.ylabel("Loss (BCE)")
    plt.title("Training Loss Evolution")
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "training_curves.png"
    else:
        save_path = Path(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves: {save_path}")


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing accuracy, F1, and ROC AUC across ansätze.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns "Ansatz", "Accuracy", "F1", "ROC_AUC".
    save_path : Path, optional
    """
    metrics = ["Accuracy", "F1", "ROC_AUC"]
    x = np.arange(len(results_df["Ansatz"]))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        label = metric.replace("_", " ").upper()
        ax.bar(x + i * width, results_df[metric], width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["Ansatz"])
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison by Ansatz")
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "metrics_comparison.png"
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics comparison: {save_path}")


def plot_structural_comparison(
    results_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing depth, size, and two-qubit gate count.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain "Ansatz", "Depth", "Gate_Count", "Two_Qubit_Gates".
    save_path : Path, optional
    """
    metrics = ["Depth", "Gate_Count", "Two_Qubit_Gates"]
    x = np.arange(len(results_df["Ansatz"]))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["Ansatz"])
    ax.set_ylabel("Value")
    ax.set_title("Structural Comparison by Ansatz")
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "structural_comparison.png"
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved structural comparison: {save_path}")


def plot_data_distribution(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Data Distribution",
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of the 2-D angular encoding used by the VQC.

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)
    y : np.ndarray of shape (n,)
    class_names : list[str], optional
    title : str
    save_path : Path, optional
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    plt.figure(figsize=(6, 5))
    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(
            X[idx, 0], X[idx, 1], label=f"{class_names[int(cls)]}", alpha=0.7
        )

    plt.xlabel("x0 (angle)")
    plt.ylabel("x1 (angle)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "data_distribution.png"
    else:
        save_path = Path(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved data distribution: {save_path}")
