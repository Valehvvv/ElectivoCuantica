"""Main entry point: end-to-end VQC ansatz comparison pipeline.

Run::

    python main.py

Modify the ``BACKEND_MODE`` variable below to switch between:

- ``"statevector"``   → exact simulation (default, no extra dependencies)
- ``"aer_simulator"`` → Qiskit Aer local simulator
- ``"ibm_simulator"`` → IBM cloud simulator
- ``"ibm_hardware"``  → real IBM Quantum device
- ``"spinq_nmr"``     → SpinQ NMR 2-qubit quantum computer
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from circuits import create_base_ansatz, create_hea_ansatz, create_reduced_ansatz
from config import (
    CLASS_NAMES,
    CLASS_PAIR,
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from dataset import save_angles_to_csv
from evaluation import compute_total_time, evaluate_model, structural_info
from ibm_runtime import get_ibm_runtime_service, select_backend
from models import ansatz_registry
from preprocessing import (
    apply_lda,
    filter_classes,
    load_dataset,
    normalize_to_pi,
    split_dataset,
)
from training import binary_cross_entropy, train_all_ansatze
from visualization import (
    plot_confusion_matrices,
    plot_data_distribution,
    plot_metrics_comparison,
    plot_structural_comparison,
    plot_training_curves,
)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION ─ change only this block to switch backends
# ──────────────────────────────────────────────────────────────────────
BACKEND_MODE: str = os.environ.get("BACKEND_MODE", "spinq_nmr")  # "statevector" | "aer_simulator" | "ibm_simulator" | "ibm_hardware"
USE_IBM_SIMULATOR: bool = True  # if "ibm_*", prefer simulator?
IBM_BACKEND_NAME: str | None = None  # explicit name or None = auto
N_SHOTS: int = 1024
MAX_ITER: int = 200


def _get_backend():
    """Resolve backend object based on ``BACKEND_MODE``."""
    if BACKEND_MODE == "statevector":
        return None

    if BACKEND_MODE == "aer_simulator":
        try:
            from qiskit_aer import AerSimulator

            return AerSimulator()
        except ImportError:
            print("[main] qiskit-aer not installed; falling back to statevector.")
            return None

    if BACKEND_MODE in ("ibm_simulator", "ibm_hardware"):
        service = get_ibm_runtime_service()
        if service is None:
            print("[main] IBM service unavailable; falling back to statevector.")
            return None
        return select_backend(
            service,
            backend_name=IBM_BACKEND_NAME,
            simulator=(BACKEND_MODE == "ibm_simulator" or USE_IBM_SIMULATOR),
        )

    if BACKEND_MODE == "spinq_nmr":
        try:
            from spinq_backend import SpinQEnvironmentError, create_spinq_backend
            from config import (
                SPINQ_IP,
                SPINQ_PASSWORD,
                SPINQ_PORT,
                SPINQ_TASK_NAME,
                SPINQ_USERNAME,
            )

            return create_spinq_backend(
                ip=SPINQ_IP,
                port=SPINQ_PORT,
                username=SPINQ_USERNAME,
                password=SPINQ_PASSWORD,
                task_name=SPINQ_TASK_NAME,
                shots=N_SHOTS,
            )
        except SpinQEnvironmentError as exc:
            print(f"[main] spinq_nmr environment check failed:\n{exc}")
            print("[main] Falling back to statevector.")
            return None
        except ImportError:
            print("[main] spinqit not installed; falling back to statevector.")
            return None

    print(f"[main] Unknown BACKEND_MODE '{BACKEND_MODE}'; using statevector.")
    return None


# ──────────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print(" PREPROCESSING")
print("=" * 60)

df_full = load_dataset()
df = filter_classes(df_full, class_pair=CLASS_PAIR)

FEATURE_NAMES_V4 = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

X_train_raw, X_test_raw, y_train, y_test = split_dataset(
    df,
    feature_names=FEATURE_NAMES_V4,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# LDA1 + Petal Width (2-D)
X_train_2d, X_test_2d, lda = apply_lda(X_train_raw, y_train, X_test_raw)

# Normalise to [0, π]
X_train_angle, X_test_angle, angle_scaler = normalize_to_pi(X_train_2d, X_test_2d)

print(f"Train shape: {X_train_angle.shape} | Test shape: {X_test_angle.shape}")
print(f"Train angle range: [{X_train_angle.min():.4f}, {X_train_angle.max():.4f}]")
print(f"Test angle range : [{X_test_angle.min():.4f}, {X_test_angle.max():.4f}]")

# Save processed data
save_angles_to_csv(X_train_angle, y_train, DATA_DIR / "train_angles.csv")
save_angles_to_csv(X_test_angle, y_test, DATA_DIR / "test_angles.csv")

# Plot data distribution
plot_data_distribution(
    X_train_angle,
    y_train,
    class_names=list(CLASS_NAMES.values()),
    title="Train data in [0, pi]",
)
plot_data_distribution(
    X_test_angle,
    y_test,
    class_names=list(CLASS_NAMES.values()),
    title="Test data in [0, pi]",
    save_path=RESULTS_DIR / "data_distribution_test.png",
)

# ──────────────────────────────────────────────────────────────────────
# 2. Resolve backend
# ──────────────────────────────────────────────────────────────────────
backend = _get_backend()

# Fallback to statevector if backend could not be resolved
actual_mode = BACKEND_MODE
if backend is None and BACKEND_MODE.startswith(("ibm", "spinq")):
    print("[main] No backend available; falling back to statevector.")
    actual_mode = "statevector"

print(f"\nBackend mode: {actual_mode}")
print(f"Backend object: {backend}")

# ──────────────────────────────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" TRAINING")
print("=" * 60)

training_results = train_all_ansatze(
    X_train_angle,
    y_train,
    backend_mode=actual_mode,
    backend=backend,
    optimizer="COBYLA",
    max_iter=MAX_ITER,
    n_shots=N_SHOTS,
)

# ──────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" EVALUATION")
print("=" * 60)

rows = []
evaluations = {}

for name, tres in training_results.items():
    vqc = tres["vqc"]
    eval_metrics = evaluate_model(vqc, X_test_angle, y_test)
    structure = structural_info(vqc, sample_x=X_train_angle[0])

    total_time = compute_total_time(
        tres["training_time_sec"], eval_metrics["inference_time_sec"]
    )

    row = {
        "Ansatz": name,
        "Backend": BACKEND_MODE,
        "Accuracy": eval_metrics["accuracy"],
        "Precision": eval_metrics["precision"],
        "Recall": eval_metrics["recall"],
        "F1": eval_metrics["f1"],
        "ROC_AUC": eval_metrics["roc_auc"],
        "Depth": structure["depth"],
        "Gate_Count": structure["size"],
        "Two_Qubit_Gates": structure["two_qubit_gates"],
        "Training_Time_sec": tres["training_time_sec"],
        "Inference_Time_sec": eval_metrics["inference_time_sec"],
        "Total_Time_sec": total_time,
        "Shots": N_SHOTS,
        "Date": datetime.now().isoformat(),
    }
    rows.append(row)
    evaluations[name] = eval_metrics

    print(f"\n{name}:")
    print(f"  Accuracy  = {eval_metrics['accuracy']:.4f}")
    print(f"  F1        = {eval_metrics['f1']:.4f}")
    print(f"  ROC AUC   = {eval_metrics['roc_auc']:.4f}")
    print(f"  Depth     = {structure['depth']}")
    print(f"  Size      = {structure['size']}")
    print(f"  2Q gates  = {structure['two_qubit_gates']}")
    print(f"  Train     = {tres['training_time_sec']:.2f} s")
    print(f"  Infer     = {eval_metrics['inference_time_sec']:.2f} s")

# ──────────────────────────────────────────────────────────────────────
# 5. Export results
# ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
results_df.to_csv(RESULTS_DIR / "results.csv", index=False)
print(f"\nResults exported to: {RESULTS_DIR / 'results.csv'}")
print(results_df.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────
# 6. Visualizations
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" VISUALIZATIONS")
print("=" * 60)

plot_training_curves(training_results)
plot_confusion_matrices(evaluations, class_names=list(CLASS_NAMES.values()))
plot_metrics_comparison(results_df)
plot_structural_comparison(results_df)

print("\nDone. All results and plots saved to:", RESULTS_DIR)
