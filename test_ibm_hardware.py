"""Quick IBM hardware test: train locally, infer on real QPU.

Trains all 3 ansätze with statevector simulation, then evaluates
only the test set predictions on ibm_fez (or other available QPU).
"""

from __future__ import annotations

import os

from ibm_runtime import get_ibm_runtime_service, select_backend
from config import CLASS_NAMES, RESULTS_DIR
from dataset import load_angles_from_csv
from evaluation import evaluate_model, structural_info
from models import VQC
from training import train_all_ansatze

DATA_DIR = RESULTS_DIR.parent / "data"

print("Loading data...")
X_train, X_test, y_train, y_test = load_angles_from_csv(
    DATA_DIR / "train_angles.csv",
    DATA_DIR / "test_angles.csv",
)

# Resolve IBM backend. IBM_FAKE=1 uses a local fake backend (Aer noise
# model from ibm_fez calibration snapshot) — full dress rehearsal of the
# EstimatorV2/ISA-transpilation path at zero QPU cost.
if os.environ.get("IBM_FAKE") == "1":
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    print("IBM_FAKE=1: using local FakeFez (no QPU jobs will be submitted)")
else:
    service = get_ibm_runtime_service()
    backend = select_backend(service) if service is not None else None
print(f"Backend: {backend}")

if backend is None:
    print("No IBM backend available. Exiting.")
    exit(1)

print("\nTraining locally (statevector)...")
results = train_all_ansatze(
    X_train,
    y_train,
    backend_mode="statevector",
    backend=None,
    max_iter=200,
)

print("\nEvaluating on IBM hardware...")
for name, tres in results.items():
    vqc_statevector = tres["vqc"]

    # Create a new VQC with IBM backend, same parameters
    vqc_ibm = VQC(
        ansatz_fn=vqc_statevector.ansatz_fn,
        n_params=vqc_statevector.n_params,
        backend_mode="ibm_hardware",
        backend=backend,
        n_shots=1024,
    )
    vqc_ibm.theta_ = vqc_statevector.theta_

    print(f"\n--- {name} on {backend.name} ---")
    metrics = evaluate_model(vqc_ibm, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"  Inference time: {metrics['inference_time_sec']:.2f} s")
    for rec in vqc_ibm.usage_log:
        print(f"  Job {rec['job_id']}: usage={rec['usage']}")

print("\nDone.")
