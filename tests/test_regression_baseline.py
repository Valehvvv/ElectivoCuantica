"""Phase 0 regression baseline test.

Runs the statevector VQC pipeline (preprocessing → training → evaluation)
using the exact same configuration as ``main.py`` and compares the
resulting Accuracy/F1 per ansatz against the frozen snapshot stored in
``results/baseline_statevector.csv``.

Purpose
-------
This test exists to catch *unintended* regressions introduced by future
changes (e.g. Phase 1's observable unification).  Because the
``statevector`` backend is fully deterministic (seeded LDA/split via
``RANDOM_STATE`` and seeded initial parameters via ``np.random.default_rng(42)``
inside ``VQC.fit``), Accuracy/F1 must match the baseline snapshot exactly
(within floating point tolerance).

If this test fails after a legitimate, intentional change to the
observable/measurement convention, regenerate the baseline deliberately
(see ``results/baseline_statevector.csv`` docstring at the top of this
file) and document *why* the numbers changed - do not just re-run and
overwrite blindly.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from config import CLASS_PAIR, FEATURE_NAMES, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from evaluation import evaluate_model
from preprocessing import apply_lda, filter_classes, load_dataset, normalize_to_pi, split_dataset
from training import train_all_ansatze

BASELINE_PATH = RESULTS_DIR / "baseline_statevector.csv"
MAX_ITER = 200
METRIC_ATOL = 1e-6


@pytest.fixture(scope="module")
def statevector_results() -> dict[str, dict[str, float]]:
    """Run the full statevector pipeline once and return metrics per ansatz."""
    df_full = load_dataset()
    df = filter_classes(df_full, class_pair=CLASS_PAIR)

    X_train_raw, X_test_raw, y_train, y_test = split_dataset(
        df,
        feature_names=FEATURE_NAMES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train_2d, X_test_2d, _lda = apply_lda(X_train_raw, y_train, X_test_raw)
    X_train_angle, X_test_angle, _scaler = normalize_to_pi(X_train_2d, X_test_2d)

    training_results = train_all_ansatze(
        X_train_angle,
        y_train,
        backend_mode="statevector",
        backend=None,
        optimizer="COBYLA",
        max_iter=MAX_ITER,
        n_shots=1024,
    )

    metrics_by_ansatz: dict[str, dict[str, float]] = {}
    for name, tres in training_results.items():
        vqc = tres["vqc"]
        eval_metrics = evaluate_model(vqc, X_test_angle, y_test)
        metrics_by_ansatz[name] = eval_metrics

    return metrics_by_ansatz


@pytest.fixture(scope="module")
def baseline_df() -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"Baseline file not found: {BASELINE_PATH}. "
            "Run the statevector pipeline once and save results/results.csv "
            "as results/baseline_statevector.csv before running this test."
        )
    return pd.read_csv(BASELINE_PATH)


@pytest.mark.parametrize("ansatz_name", ["Base", "Reducido", "HEA"])
def test_accuracy_matches_baseline(statevector_results, baseline_df, ansatz_name):
    baseline_row = baseline_df.loc[baseline_df["Ansatz"] == ansatz_name].iloc[0]
    expected_accuracy = float(baseline_row["Accuracy"])
    actual_accuracy = statevector_results[ansatz_name]["accuracy"]

    assert math.isclose(actual_accuracy, expected_accuracy, abs_tol=METRIC_ATOL), (
        f"{ansatz_name}: Accuracy regressed from baseline "
        f"({expected_accuracy} -> {actual_accuracy})"
    )


@pytest.mark.parametrize("ansatz_name", ["Base", "Reducido", "HEA"])
def test_f1_matches_baseline(statevector_results, baseline_df, ansatz_name):
    baseline_row = baseline_df.loc[baseline_df["Ansatz"] == ansatz_name].iloc[0]
    expected_f1 = float(baseline_row["F1"])
    actual_f1 = statevector_results[ansatz_name]["f1"]

    assert math.isclose(actual_f1, expected_f1, abs_tol=METRIC_ATOL), (
        f"{ansatz_name}: F1 regressed from baseline ({expected_f1} -> {actual_f1})"
    )
