"""Variational Quantum Classifier (VQC) model class.

Supports multiple backends via a clean abstraction:

- ``"statevector"``      → exact simulation (Qiskit Statevector)
- ``"aer_simulator"``    → Qiskit Aer simulator
- ``"ibm_simulator"``    → IBM Quantum cloud simulator
- ``"ibm_hardware"``     → IBM Quantum real device
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit

from backends import QuantumBackend, get_backend
from config import DEFAULT_SHOTS
from utils import clamp_probabilities

AnsatzBuilder = Callable[..., QuantumCircuit]


class VQC:
    """Variational Quantum Classifier for 2-qubit binary classification.

    Parameters
    ----------
    ansatz_fn : callable
        Function ``ansatz_fn(x, theta) -> QuantumCircuit``.
    n_params : int
        Number of trainable parameters (length of ``theta``).
    backend_mode : str
        One of ``"statevector"``, ``"aer_simulator"``, ``"ibm_simulator"``,
        ``"ibm_hardware"``.
    backend : optional
        Pre-configured Qiskit backend instance.  When ``None`` for
        ``"statevector"`` mode, exact statevector simulation is used.
    optimizer : str
        Optimiser name (passed through to ``scipy.optimize.minimize``).
    n_shots : int
        Number of measurement shots for non-statevector backends.
    quantum_backend : backends.QuantumBackend, optional
        Pre-built backend adapter (see ``backends.py``). When provided,
        it takes precedence over ``backend_mode``/``backend``/``n_shots``
        — mainly useful for dependency injection in tests (e.g. a fake
        backend that counts batch calls). When ``None`` (the default), a
        concrete backend is resolved from ``backend_mode`` via
        ``backends.get_backend``.

    Attributes
    ----------
    theta_ : np.ndarray or None
        Learned parameters after ``fit()`` is called.
    history_ : list[float]
        Loss value at each function evaluation.
    training_time_ : float
        Seconds spent in ``fit()``.
    """

    def __init__(
        self,
        ansatz_fn: AnsatzBuilder,
        n_params: int,
        backend_mode: str = "statevector",
        backend: Any = None,
        optimizer: str = "COBYLA",
        n_shots: int = DEFAULT_SHOTS,
        quantum_backend: QuantumBackend | None = None,
    ) -> None:
        self.ansatz_fn = ansatz_fn
        self.n_params = n_params
        self.backend_mode = backend_mode
        self.backend = backend
        self.optimizer = optimizer
        self.n_shots = n_shots

        self._quantum_backend: QuantumBackend = (
            quantum_backend
            if quantum_backend is not None
            else get_backend(backend_mode, backend, n_shots)
        )

        self.theta_: np.ndarray | None = None
        self.history_: list[float] = []
        self.training_time_: float = 0.0

    # ------------------------------------------------------------------
    # Internal: batched expectation value computation
    # ------------------------------------------------------------------
    def _expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        """Compute the classifier's readout ``<Z>`` for a batch of circuits.

        Delegates to the injected/resolved ``QuantumBackend``
        (``backends.py``), which targets a single, fixed physical qubit
        (``observable.MEASURED_QUBIT_INDEX``) consistently across every
        backend implementation — see ``docs/observable_convention.md``.

        All circuits needed for one training iteration or one prediction
        call are passed in a single list so that each concrete backend
        can submit them as one batch (Phase 4), instead of one
        circuit/job per sample.
        """
        return self._quantum_backend.expectations(circuits)

    # ------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted class-1 probabilities for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 2)
            Input features.

        Returns
        -------
        probs : np.ndarray of shape (n_samples,)
        """
        if self.theta_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        circuits = [self.ansatz_fn(list(x), self.theta_) for x in X]
        exps = self._expectations(circuits)  # single batched backend call
        probs = [clamp_probabilities((exp + 1.0) / 2.0) for exp in exps]

        return np.array(probs, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions (0/1) via threshold 0.5.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 2)

        Returns
        -------
        pred : np.ndarray of shape (n_samples,)
        """
        return (self.predict_proba(X) > 0.5).astype(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_theta: np.ndarray | None = None,
        max_iter: int = 200,
    ) -> VQC:
        """Fit the VQC model via the configured classical optimiser.

        Parameters
        ----------
        X : np.ndarray of shape (n, 2)
        y : np.ndarray of shape (n,)
        initial_theta : np.ndarray or None
            Initial parameters; if ``None``, uniform random in
            ``[-0.1, 0.1]``.
        max_iter : int
            Maximum iterations for the optimiser.

        Returns
        -------
        self
        """
        from config import BCE_EPS

        self.history_ = []

        def loss(theta: np.ndarray) -> float:
            circuits = [self.ansatz_fn(list(x_i), theta) for x_i in X]
            exps = self._expectations(circuits)  # single batched backend call
            probs_arr = np.array(
                [clamp_probabilities((exp + 1.0) / 2.0, eps=BCE_EPS) for exp in exps]
            )
            bce = -np.mean(y * np.log(probs_arr) + (1.0 - y) * np.log(1.0 - probs_arr))
            self.history_.append(float(bce))
            return float(bce)

        if initial_theta is None:
            initial_theta = np.random.default_rng(42).uniform(
                -0.1, 0.1, self.n_params
            )

        from scipy.optimize import minimize

        t0 = time.perf_counter()
        result = minimize(
            loss,
            initial_theta,
            method=self.optimizer,
            options={"maxiter": max_iter},
        )
        self.training_time_ = time.perf_counter() - t0

        self.theta_ = result.x
        return self


def ansatz_registry() -> dict[str, tuple[AnsatzBuilder, int]]:
    """Return ``{name: (builder_fn, n_params)}`` for the three ansätze."""
    from circuits import create_base_ansatz, create_hea_ansatz, create_reduced_ansatz

    return {
        "Base": (create_base_ansatz, 6),
        "Reducido": (create_reduced_ansatz, 4),
        "HEA": (create_hea_ansatz, 4),
    }
