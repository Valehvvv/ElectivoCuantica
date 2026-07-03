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

import importlib.metadata
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from circuits import create_base_ansatz, create_hea_ansatz, create_reduced_ansatz
from config import (
    BACKEND_ENDIANNESS,
    CLASS_NAMES,
    CLASS_PAIR,
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from dataset import save_angles_to_csv
from evaluation import compute_total_time, evaluate_model, structural_info
from ibm_runtime import get_ibm_runtime_service, select_backend
from models import ansatz_registry
from observability import git_commit_hash, init_run
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

_RUN_START = time.perf_counter()

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION ─ change only this block to switch backends
# ──────────────────────────────────────────────────────────────────────
BACKEND_MODE: str = os.environ.get("BACKEND_MODE", "spinq_nmr")  # "statevector" | "aer_simulator" | "ibm_simulator" | "ibm_hardware"
USE_IBM_SIMULATOR: bool = True  # if "ibm_*", prefer simulator?
IBM_BACKEND_NAME: str | None = None  # explicit name or None = auto

# The following overrides exist for quick experimentation / short test
# runs without editing this file; all default to the previous hard-coded
# values so behaviour is unchanged when the variables are unset.
N_SHOTS: int = int(os.environ.get("N_SHOTS", "1024"))
MAX_ITER: int = int(os.environ.get("MAX_ITER", "200"))
N_TRAIN: int | None = (
    int(os.environ.get("N_TRAIN")) if os.environ.get("N_TRAIN") else None
)  # if set, use a stratified/class-balanced subsample of size N_TRAIN from
# the training set (seeded with RANDOM_STATE for reproducibility) instead
# of the full training set
_ANSATZ_ENV = os.environ.get("ANSATZ")  # comma-separated names, e.g. "HEA,Base"
ANSATZ_FILTER: list[str] | None = (
    [a.strip() for a in _ANSATZ_ENV.split(",") if a.strip()] if _ANSATZ_ENV else None
)

run = init_run(BACKEND_MODE)
log = run.logger


def _get_backend():
    """Resolve backend object based on ``BACKEND_MODE``."""
    if BACKEND_MODE == "statevector":
        return None

    if BACKEND_MODE == "aer_simulator":
        try:
            from qiskit_aer import AerSimulator

            return AerSimulator()
        except ImportError:
            log.warning("qiskit-aer not installed; falling back to statevector.")
            return None

    if BACKEND_MODE in ("ibm_simulator", "ibm_hardware"):
        service = get_ibm_runtime_service()
        if service is None:
            log.warning("IBM service unavailable; falling back to statevector.")
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
            log.warning("spinq_nmr environment check failed: %s", exc)
            log.warning("Falling back to statevector.")
            return None
        except ImportError:
            log.warning("spinqit not installed; falling back to statevector.")
            return None

    log.warning("Unknown BACKEND_MODE '%s'; using statevector.", BACKEND_MODE)
    return None


# ──────────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ──────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(" PREPROCESSING")
log.info("=" * 60)

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

# Optional stratified subsample of the training set (N_TRAIN override, see
# CONFIGURATION block above). Keeps class balance and stays reproducible
# via RANDOM_STATE.
if N_TRAIN is not None and N_TRAIN < len(X_train_angle):
    from sklearn.model_selection import train_test_split as _tts

    X_train_angle, _, y_train, _ = _tts(
        X_train_angle,
        y_train,
        train_size=N_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    log.info(
        "N_TRAIN override applied: using %d/%d stratified training samples",
        len(X_train_angle),
        N_TRAIN,
    )

log.info("Train shape: %s | Test shape: %s", X_train_angle.shape, X_test_angle.shape)
log.info(
    "Train angle range: [%.4f, %.4f]",
    X_train_angle.min(),
    X_train_angle.max(),
)
log.info(
    "Test angle range : [%.4f, %.4f]",
    X_test_angle.min(),
    X_test_angle.max(),
)

# Save processed data
save_angles_to_csv(X_train_angle, y_train, DATA_DIR / "train_angles.csv")
save_angles_to_csv(X_test_angle, y_test, DATA_DIR / "test_angles.csv")

# Plot data distribution
plot_data_distribution(
    X_train_angle,
    y_train,
    class_names=list(CLASS_NAMES.values()),
    title="Train data in [0, pi]",
    save_path=run.results_path("data_distribution.png"),
)
plot_data_distribution(
    X_test_angle,
    y_test,
    class_names=list(CLASS_NAMES.values()),
    title="Test data in [0, pi]",
    save_path=run.results_path("data_distribution_test.png"),
)

# ──────────────────────────────────────────────────────────────────────
# 2. Resolve backend
# ──────────────────────────────────────────────────────────────────────
backend = _get_backend()

# Fallback to statevector if backend could not be resolved
actual_mode = BACKEND_MODE
if backend is None and BACKEND_MODE.startswith(("ibm", "spinq")):
    log.warning("No backend available; falling back to statevector.")
    actual_mode = "statevector"

log.info("Backend mode: %s", actual_mode)
log.info("Backend object: %s", backend)

# ──────────────────────────────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(" TRAINING")
log.info("=" * 60)

full_registry = ansatz_registry()
if ANSATZ_FILTER is not None:
    wanted = {a.lower() for a in ANSATZ_FILTER}
    training_registry = {
        name: spec for name, spec in full_registry.items() if name.lower() in wanted
    }
    if not training_registry:
        raise ValueError(
            f"ANSATZ={ANSATZ_FILTER!r} matched none of the known ansätze "
            f"{list(full_registry)!r}"
        )
    log.info("ANSATZ override applied: training only %s", list(training_registry))
else:
    training_registry = full_registry

training_results = train_all_ansatze(
    X_train_angle,
    y_train,
    backend_mode=actual_mode,
    backend=backend,
    optimizer="COBYLA",
    max_iter=MAX_ITER,
    n_shots=N_SHOTS,
    registry=training_registry,
)

# ──────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(" EVALUATION")
log.info("=" * 60)

rows = []
evaluations = {}
prediction_frames = []

for name, tres in training_results.items():
    vqc = tres["vqc"]
    eval_metrics = evaluate_model(vqc, X_test_angle, y_test)
    structure = structural_info(vqc, sample_x=X_train_angle[0])

    total_time = compute_total_time(
        tres["training_time_sec"], eval_metrics["inference_time_sec"]
    )

    row = {
        "Ansatz": name,
        "Backend": actual_mode,
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

    # Per-sample predictions for this ansatz (expectation_z is <Z> in
    # [-1, 1]; prob is the sigmoid-like mapping used by VQC.predict_proba).
    probs = vqc.predict_proba(X_test_angle)
    preds = (probs > 0.5).astype(int)
    prediction_frames.append(
        pd.DataFrame(
            {
                "ansatz": name,
                "y_true": y_test,
                "y_pred": preds,
                "prob": probs,
                "expectation_z": 2.0 * probs - 1.0,
            }
        )
    )

    log.info(
        "%s: accuracy=%.4f f1=%.4f roc_auc=%.4f depth=%d size=%d two_qubit_gates=%d "
        "train_time=%.2fs infer_time=%.2fs",
        name,
        eval_metrics["accuracy"],
        eval_metrics["f1"],
        eval_metrics["roc_auc"],
        structure["depth"],
        structure["size"],
        structure["two_qubit_gates"],
        tres["training_time_sec"],
        eval_metrics["inference_time_sec"],
    )

predictions_df = pd.concat(prediction_frames, ignore_index=True)
run.save_predictions(predictions_df)
log.info("Predictions saved to: %s", run.results_path("predictions.csv"))

# ──────────────────────────────────────────────────────────────────────
# 5. Export results
# ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
results_path = run.results_path("results.csv")
results_df.to_csv(results_path, index=False)
log.info("Results exported to: %s", results_path)
log.info("\n%s", results_df.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────
# 6. Visualizations
# ──────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(" VISUALIZATIONS")
log.info("=" * 60)

plot_training_curves(training_results, save_path=run.results_path("training_curves.png"))
plot_confusion_matrices(
    evaluations,
    class_names=list(CLASS_NAMES.values()),
    save_path=run.results_path("confusion_matrices.png"),
)
plot_metrics_comparison(results_df, save_path=run.results_path("metrics_comparison.png"))
plot_structural_comparison(results_df, save_path=run.results_path("structural_comparison.png"))

# ──────────────────────────────────────────────────────────────────────
# 7. Metadata
# ──────────────────────────────────────────────────────────────────────
_duration_sec = time.perf_counter() - _RUN_START

_spinqit_version = None
if actual_mode == "spinq_nmr":
    try:
        _spinqit_version = importlib.metadata.version("spinqit")
    except importlib.metadata.PackageNotFoundError:
        _spinqit_version = None

run.write_metadata(
    {
        "run_id": run.run_id,
        "backend_mode": BACKEND_MODE,
        "actual_mode": actual_mode,
        "MAX_ITER": MAX_ITER,
        "N_SHOTS": N_SHOTS,
        "n_train": int(len(X_train_angle)),
        "n_test": int(len(X_test_angle)),
        "RANDOM_STATE": RANDOM_STATE,
        "ansatze": list(training_registry),
        "endianness": BACKEND_ENDIANNESS.get(actual_mode),
        "git_commit": git_commit_hash(),
        "spinqit_version": _spinqit_version,
        "duracion_total_seg": _duration_sec,
    }
)
log.info("Metadata written to: %s", run.results_path("run_metadata.json"))

log.info("Done. All results and plots saved to: %s", run.run_dir)
