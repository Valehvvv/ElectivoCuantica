"""Tests for the backend contract (Phase 5).

Verifies that:

1. ``QuantumBackend`` is an abstract base class with the required ``expectations`` method.
2. ``StatevectorBackend`` correctly implements the contract and matches the expected
   Pauli string convention (see ``observable.py`` / ``docs/observable_convention.md``).
3. ``BackendError`` is raised appropriately for invalid backend configurations.
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp

from backends.contract import BackendError, QuantumBackend, StatevectorBackend
from config import OBSERVABLE_PAULI


# ---------------------------------------------------------------------------
# Abstract base class tests
# ---------------------------------------------------------------------------
def test_quantum_backend_is_abstract():
    """Ensure QuantumBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        QuantumBackend()  # type: ignore


def test_quantum_backend_requires_expectations():
    """Ensure subclasses must implement expectations."""
    class IncompleteBackend(QuantumBackend):
        pass

    with pytest.raises(TypeError):
        IncompleteBackend()


# ---------------------------------------------------------------------------
# StatevectorBackend tests
# ---------------------------------------------------------------------------
def test_statevector_backend_initializes_observable():
    """StatevectorBackend should initialize with the correct observable."""
    backend = StatevectorBackend()
    assert isinstance(backend._observable, SparsePauliOp)
    # The observable should match the configured Pauli string
    expected = SparsePauliOp.from_list([(OBSERVABLE_PAULI, 1)])
    assert backend._observable == expected


def test_statevector_backend_expectations_returns_list():
    """StatevectorBackend.expectations should return a list of floats."""
    backend = StatevectorBackend()
    qc = QuantumCircuit(2)
    qc.ry(0.5, 0)
    qc.cx(0, 1)

    result = backend.expectations([qc])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], float)


def test_statevector_backend_expectations_matches_manual_calculation():
    """StatevectorBackend should match manual expectation value calculation."""
    backend = StatevectorBackend()

    # Create a simple circuit
    qc = QuantumCircuit(2)
    qc.x(0)  # X on qubit 0

    # Manual calculation
    sv = Statevector.from_instruction(qc)
    from qiskit.quantum_info import Pauli
    expected = float(sv.expectation_value(Pauli(OBSERVABLE_PAULI)))

    actual = backend.expectations([qc])[0]

    assert actual == pytest.approx(expected)


def test_statevector_backend_expectations_order_preserved():
    """StatevectorBackend should preserve the order of input circuits."""
    backend = StatevectorBackend()

    qc1 = QuantumCircuit(2)
    qc1.ry(0.3, 0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.x(1)

    results = backend.expectations([qc1, qc2])
    assert len(results) == 2
    # Results should be in the same order as input circuits
    assert results[0] != results[1]  # Different circuits should give different results


# ---------------------------------------------------------------------------
# BackendError tests
# ---------------------------------------------------------------------------
def test_backend_error_is_exception():
    """BackendError should be a subclass of Exception."""
    error = BackendError("test")
    assert isinstance(error, Exception)


def test_backend_error_message_preserved():
    """BackendError should preserve the error message."""
    message = "test error message"
    error = BackendError(message)
    assert str(error) == message