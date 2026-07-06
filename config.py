"""Central configuration for the VQC ansatz comparison project."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _load_env() -> None:
    """Load environment variables from ``.env`` file if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
CLASS_PAIR: tuple[int, int] = (1, 2)
CLASS_NAMES: dict[int, str] = {0: "versicolor", 1: "virginica"}

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
ANGLE_RANGE: tuple[float, float] = (0.0, float(np.pi))

FEATURE_NAMES: list[str] = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# ---------------------------------------------------------------------------
# Quantum / backend defaults
# ---------------------------------------------------------------------------
N_QUBITS: int = 2
DEFAULT_BACKEND: str = "statevector"  # "statevector" | "aer_simulator" | "ibm_simulator" | "ibm_hardware"
DEFAULT_OPTIMIZER: str = "COBYLA"
DEFAULT_MAX_ITER: int = 200
DEFAULT_SHOTS: int = 1024

# ---------------------------------------------------------------------------
# Observable / readout convention (see docs/observable_convention.md)
# ---------------------------------------------------------------------------
# The classifier reads out Pauli-Z on physical circuit qubit index
# ``observable.MEASURED_QUBIT_INDEX`` (currently 1), not qubit 0.  This
# matches Qiskit's little-endian convention where the "ZI" Pauli string
# acts on qubit 1 (leftmost char = highest qubit index) and where
# ``counts`` bitstrings have ``bitstring[0]`` == qubit 1.  This convention
# was kept as-is (rather than "corrected" to the real qubit 0) specifically
# to avoid changing the statevector baseline captured in
# ``results/baseline_statevector.csv`` (Phase 0).  ``OBSERVABLE_PAULI``
# below MUST stay consistent with ``observable.MEASURED_QUBIT_INDEX``.
OBSERVABLE_PAULI: str = "ZI"

# Bitstring ordering ("endianness") per backend mode, consumed by
# ``observable.expectation_z_qubit0_from_counts`` inside
# ``models.VQC._expectation``.  All Qiskit-based backends use "little"
# (bitstring[0] == qubit n-1).  SpinQ NMR is assumed "big" (bitstring[0]
# == qubit 0) based on how ``spinq_backend._probabilities_to_counts``
# constructs its bitstrings from the ``[p00, p01, p10, p11]`` probability
# list -- TODO: verify this against real SpinQ NMR hardware output before
# trusting SpinQ results quantitatively.
BACKEND_ENDIANNESS: dict[str, str] = {
    "statevector": "little",  # not actually used for counts (analytic path)
    "aer_simulator": "little",
    "ibm_simulator": "little",
    "ibm_hardware": "little",
    "spinq_nmr": "big",  # verified: "little" gave acc 0.00 vs "big" 0.35
    "qryd": "little",  # QRydDemo uses Qiskit's standard little-endian
}
DEFAULT_ENDIANNESS: str = "little"

# ---------------------------------------------------------------------------
# IBM Quantum credentials (set via environment or override here)
# ---------------------------------------------------------------------------
IBMQ_TOKEN: str = os.environ.get("IBMQ_TOKEN", "")
IBMQ_INSTANCE: str = os.environ.get("IBMQ_INSTANCE", "")  # dejar vacío para plan Open
IBMQ_BACKEND_SIMULATOR: str = "ibmq_qasm_simulator"
IBMQ_BACKEND_HARDWARE: str = "ibm_fez"

# ---------------------------------------------------------------------------
# SpinQ NMR configuration
# ---------------------------------------------------------------------------
SPINQ_IP: str = os.environ.get("SPINQ_IP", "")
SPINQ_PORT: int = int(os.environ.get("SPINQ_PORT", "8989"))
SPINQ_USERNAME: str = os.environ.get("SPINQ_USERNAME", "")
SPINQ_PASSWORD: str = os.environ.get("SPINQ_PASSWORD", "")
SPINQ_TASK_NAME: str = os.environ.get("SPINQ_TASK_NAME", "VQC-Experiment")

# ---------------------------------------------------------------------------
# QRydDemo configuration
# ---------------------------------------------------------------------------
QRYD_TOKEN: str = os.environ.get("QRYD_API_TOKEN", "")
QRYD_BACKEND_NAME: str = os.environ.get("QRYD_BACKEND_NAME", "qryd_emulator$square")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
INITIAL_THETA_RANGE: tuple[float, float] = (-0.1, 0.1)
BCE_EPS: float = 1e-10
