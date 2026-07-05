from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp

from config import OBSERVABLE_PAULI


class BackendError(Exception):
    """Base exception for backend-related errors."""
    pass


class QuantumBackend(ABC):
    """Common interface every concrete backend must implement.

    Implementations must be stateless with respect to circuits (no
    caching between calls is assumed) and must return values in the same
    order as the input ``circuits`` list.
    """

    @abstractmethod
    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        """Return the readout ``<Z>`` value for each circuit, in order.

        Parameters
        ----------
        circuits : list[QuantumCircuit]
            Circuits **without** measurements/observables attached; each
            backend adds whatever it needs (``measure_all()``, an
            ``EstimatorV2`` PUB observable, etc.).

        Returns
        -------
        list[float]
            One value in ``[-1, 1]`` per input circuit.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Statevector (exact, analytic)
# ---------------------------------------------------------------------------
class StatevectorBackend(QuantumBackend):
    """Exact statevector simulation via ``Statevector`` + ``SparsePauliOp``.

    Behaviourally identical to the pre-Phase-3 ``statevector`` branch of
    ``models.VQC._expectation`` — this preserves the frozen regression
    baseline (``results/baseline_statevector.csv``).
    """

    def __init__(self) -> None:
        self._observable = SparsePauliOp.from_list([(OBSERVABLE_PAULI, 1)])

    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        return [
            float(
                np.real(
                    Statevector.from_instruction(qc).expectation_value(
                        self._observable
                    )
                )
            )
            for qc in circuits
        ]