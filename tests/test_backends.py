"""Phase 3 / Phase 4 tests: backend abstraction + batching.

Verifies that:

1. ``backends.get_backend`` (the factory) resolves the correct concrete
   ``QuantumBackend`` subclass for each ``backend_mode`` string, without
   ``VQC`` ever inspecting those strings itself (Phase 3).
2. ``VQC.fit`` / ``VQC.predict_proba`` submit **one** batched call to the
   injected backend per loss-evaluation / prediction, containing *all*
   samples at once — never one call per sample (Phase 4). This is
   verified with a ``FakeBackend`` that records every call it receives,
   standing in for a real IBM/SpinQ backend without requiring hardware
   or credentials.
3. ``StatevectorBackend`` (used behind the scenes for ``"statevector"``)
   still reproduces the exact analytic ``<Z>`` expectation value used by
   the pre-Phase-3 code path, preserving the regression baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp

from backends import (
    AerBackend,
    IBMBackend,
    QuantumBackend,
    SpinQBackend,
    StatevectorBackend as ContractStatevectorBackend,
    get_backend,
)
from circuits import create_reduced_ansatz
from config import OBSERVABLE_PAULI
from models import VQC


# ---------------------------------------------------------------------------
# Test double: records every batch call it receives.
# ---------------------------------------------------------------------------
class FakeBackend(QuantumBackend):
    """Minimal ``QuantumBackend`` that records call/batch-size history.

    Returns a deterministic (but arbitrary) ``<Z>`` value per circuit so
    that ``VQC.fit``/``predict_proba`` run to completion without any
    quantum simulation at all.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.batch_sizes: list[int] = []

    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        self.call_count += 1
        self.batch_sizes.append(len(circuits))
        # Deterministic values in [-1, 1], independent of circuit content.
        return [0.0 for _ in circuits]


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------
def test_factory_statevector_mode():
    backend = get_backend("statevector")
    assert isinstance(backend, ContractStatevectorBackend)


def test_factory_aer_mode():
    fake_qiskit_backend = object()
    backend = get_backend("aer_simulator", backend=fake_qiskit_backend, n_shots=256)
    assert isinstance(backend, AerBackend)


def test_factory_ibm_simulator_mode():
    fake_ibm_backend = object()
    backend = get_backend("ibm_simulator", backend=fake_ibm_backend)
    assert isinstance(backend, IBMBackend)


def test_factory_ibm_hardware_mode():
    fake_ibm_backend = object()
    backend = get_backend("ibm_hardware", backend=fake_ibm_backend)
    assert isinstance(backend, IBMBackend)


def test_factory_spinq_mode():
    fake_spinq_backend = object()
    backend = get_backend("spinq_nmr", backend=fake_spinq_backend)
    assert isinstance(backend, SpinQBackend)


def test_factory_unknown_mode_falls_back_to_aer_style():
    """Unknown modes fall back to the generic counts-based backend, matching
    the previous ``models.VQC._expectation`` fallback behaviour."""
    fake_backend = object()
    backend = get_backend("some_future_backend", backend=fake_backend)
    assert isinstance(backend, AerBackend)


def test_factory_requires_backend_instance_for_non_statevector():
    with pytest.raises(ValueError):
        get_backend("aer_simulator", backend=None)
    with pytest.raises(ValueError):
        get_backend("ibm_simulator", backend=None)
    with pytest.raises(ValueError):
        get_backend("spinq_nmr", backend=None)


# ---------------------------------------------------------------------------
# VQC never sees backend_mode strings: dependency injection via
# `quantum_backend` fully bypasses the factory.
# ---------------------------------------------------------------------------
def _make_vqc(fake_backend: FakeBackend) -> VQC:
    return VQC(
        ansatz_fn=create_reduced_ansatz,
        n_params=4,
        quantum_backend=fake_backend,
    )


def test_fit_makes_one_batched_call_per_loss_evaluation():
    fake_backend = FakeBackend()
    vqc = _make_vqc(fake_backend)

    n_samples = 7
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(n_samples, 2))
    y = rng.integers(0, 2, size=n_samples)

    vqc.fit(X, y, max_iter=10)

    # COBYLA may evaluate the loss more than once per "iteration", but
    # every evaluation must be exactly ONE batched backend call...
    assert fake_backend.call_count == len(vqc.history_)
    assert fake_backend.call_count > 0
    # ...covering ALL samples at once, never per-sample calls.
    assert all(size == n_samples for size in fake_backend.batch_sizes)


def test_predict_proba_makes_a_single_batched_call():
    fake_backend = FakeBackend()
    vqc = _make_vqc(fake_backend)
    vqc.theta_ = np.zeros(4)

    n_samples = 11
    X = np.random.default_rng(1).uniform(0, np.pi, size=(n_samples, 2))

    fake_backend.call_count = 0
    fake_backend.batch_sizes = []
    probs = vqc.predict_proba(X)

    assert fake_backend.call_count == 1
    assert fake_backend.batch_sizes == [n_samples]
    assert probs.shape == (n_samples,)


def test_number_of_backend_calls_independent_of_sample_count():
    """O(1) submissions per iteration/prediction w.r.t. n_samples (Phase 4
    acceptance criterion): doubling the sample count must NOT change the
    number of backend calls, only the batch size."""
    small_backend = FakeBackend()
    vqc_small = _make_vqc(small_backend)
    vqc_small.theta_ = np.zeros(4)
    X_small = np.zeros((3, 2))
    vqc_small.predict_proba(X_small)

    large_backend = FakeBackend()
    vqc_large = _make_vqc(large_backend)
    vqc_large.theta_ = np.zeros(4)
    X_large = np.zeros((300, 2))
    vqc_large.predict_proba(X_large)

    assert small_backend.call_count == large_backend.call_count == 1


# ---------------------------------------------------------------------------
# VQC construction without dependency injection still resolves a backend
# via the factory from backend_mode/backend/n_shots (back-compat).
# ---------------------------------------------------------------------------
def test_vqc_default_construction_resolves_statevector_backend():
    vqc = VQC(ansatz_fn=create_reduced_ansatz, n_params=4)
    assert isinstance(vqc._quantum_backend, ContractStatevectorBackend)


def test_vqc_construction_resolves_backend_from_mode_string():
    fake_qiskit_backend = object()
    vqc = VQC(
        ansatz_fn=create_reduced_ansatz,
        n_params=4,
        backend_mode="spinq_nmr",
        backend=fake_qiskit_backend,
    )
    assert isinstance(vqc._quantum_backend, SpinQBackend)


# ---------------------------------------------------------------------------
# StatevectorBackend preserves the exact pre-Phase-3 analytic computation.
# ---------------------------------------------------------------------------
def test_statevector_backend_matches_manual_expectation_value():
    observable = SparsePauliOp.from_list([(OBSERVABLE_PAULI, 1)])

    qc1 = QuantumCircuit(2)
    qc1.ry(0.3, 0)
    qc1.ry(1.1, 1)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.x(1)

    expected = [
        float(np.real(Statevector.from_instruction(qc).expectation_value(observable)))
        for qc in (qc1, qc2)
    ]

    backend = ContractStatevectorBackend()
    actual = backend.expectations([qc1, qc2])

    assert actual == pytest.approx(expected)
