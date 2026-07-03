"""Central configuration for the VQC ansatz comparison project."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

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

# Observable: Pauli-Z on qubit 0
OBSERVABLE_PAULI: str = "ZI"

# ---------------------------------------------------------------------------
# IBM Quantum credentials (set via environment or override here)
# ---------------------------------------------------------------------------
IBMQ_TOKEN: str = os.environ.get("IBMQ_TOKEN", "")
IBMQ_INSTANCE: str = os.environ.get("IBMQ_INSTANCE", "ibm-q/open/main")
IBMQ_BACKEND_SIMULATOR: str = "ibmq_qasm_simulator"
IBMQ_BACKEND_HARDWARE: str = "ibm_brisbane"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
INITIAL_THETA_RANGE: tuple[float, float] = (-0.1, 0.1)
BCE_EPS: float = 1e-10
