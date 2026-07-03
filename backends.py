"""Backend abstraction layer for the VQC readout (Phases 3-4).

Problem this module solves
---------------------------
Before this module existed, ``models.VQC._expectation`` contained a chain
of ``if backend_mode == ...`` branches (statevector / spinq_nmr / ibm* /
Aer-fallback), one circuit at a time.  That coupled ``VQC`` to every
concrete backend's execution API and made "add a backend" mean "edit
``VQC``". It also meant one circuit == one hardware/job submission, which
is unusable on IBM real hardware (``n_samples x maxiter`` queued jobs).

Design
------
``QuantumBackend`` is the single interface ``VQC`` depends on. It exposes
one method, :meth:`QuantumBackend.expectations`, which takes the *entire*
batch of circuits needed for one training iteration or one prediction
call and returns the readout ``<Z>`` expectation value (see
``observable.py`` / ``docs/observable_convention.md``) for each circuit,
in order.

Every concrete backend decides internally how many actual submissions
that requires; the goal of Phase 4 is exactly **one** submission per call
whenever the execution model allows it:

- ``StatevectorBackend`` : analytic, no submission at all.
- ``AerBackend``          : one ``backend.run(list_of_circuits, ...)`` call.
- ``IBMBackend``          : one ``EstimatorV2.run(pubs)`` call (one PUB per
  circuit).
- ``SpinQBackend``        : one ``SpinQNMRBackend.run(list_of_circuits)``
  call (that method already accepts a list).

Adding a new backend therefore means: one new ``QuantumBackend``
subclass + one branch in :func:`get_backend`. ``VQC`` itself never sees
backend-mode strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp

from config import BACKEND_ENDIANNESS, DEFAULT_ENDIANNESS, DEFAULT_SHOTS, OBSERVABLE_PAULI
from observable import expectation_z_qubit0_from_counts


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


# ---------------------------------------------------------------------------
# Aer / generic Qiskit-style counts backend
# ---------------------------------------------------------------------------
class AerBackend(QuantumBackend):
    """Local/legacy Qiskit-compatible backend driven by measurement counts.

    Works with anything exposing ``backend.run(circuits, shots=...)`` that
    returns a ``Result`` with per-circuit ``get_counts(i)`` — this is
    Aer's own contract, and also the fallback used previously for any
    "legacy" backend that isn't SpinQ or IBM Runtime.

    All circuits passed to :meth:`expectations` are submitted in a
    **single** ``run()`` call (Phase 4 batching); trivial for a local
    simulator but keeps the interface's O(1)-submissions contract
    uniform across backends.
    """

    def __init__(
        self,
        backend: Any,
        n_shots: int = DEFAULT_SHOTS,
        endianness: str = "little",
    ) -> None:
        if backend is None:
            raise ValueError("AerBackend requires a configured Qiskit backend instance.")
        self._backend = backend
        self._n_shots = n_shots
        self._endianness = endianness

    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        if not circuits:
            return []

        measured = [qc.copy() for qc in circuits]
        for qc in measured:
            qc.measure_all()

        job = self._backend.run(measured, shots=self._n_shots)
        result = job.result()
        return [
            expectation_z_qubit0_from_counts(result.get_counts(i), self._endianness)
            for i in range(len(measured))
        ]


# ---------------------------------------------------------------------------
# IBM Quantum Runtime (EstimatorV2)
# ---------------------------------------------------------------------------
class IBMBackend(QuantumBackend):
    """IBM Quantum Runtime backend using ``EstimatorV2``.

    Every circuit is transpiled to the backend's ISA (via
    ``generate_preset_pass_manager``) and the shared readout observable
    (``config.OBSERVABLE_PAULI``) is remapped onto the transpiled physical
    layout with ``SparsePauliOp.apply_layout`` before estimation — this is
    required because ``EstimatorV2`` needs observables expressed in the
    *transpiled* circuit's qubit indices.

    All circuits for one call are packed into **one** ``EstimatorV2.run``
    invocation (one PUB per circuit), so one COBYLA iteration = one IBM
    Runtime job, regardless of the number of samples.

    Note
    ----
    This class could not be exercised against a real IBM Runtime service
    in this environment (no credentials / hardware access). The
    ``EstimatorV2`` PUB-result contract (``pub_result.data.evs`` as a
    scalar ``ndarray`` per PUB) was verified locally against
    ``qiskit.primitives.StatevectorEstimator``, which implements the same
    Estimator V2 primitive protocol that ``qiskit_ibm_runtime.EstimatorV2``
    documents itself as following. The transpile + ``apply_layout`` step
    was verified against a local fake backend
    (``qiskit_ibm_runtime.fake_provider.FakeManilaV2``). The actual job
    submission/queueing behaviour on real hardware was not verified here.
    """

    def __init__(self, backend: Any, n_shots: int = DEFAULT_SHOTS) -> None:
        if backend is None:
            raise ValueError("IBMBackend requires a configured IBM backend instance.")
        self._backend = backend
        self._n_shots = n_shots
        self._observable = SparsePauliOp.from_list([(OBSERVABLE_PAULI, 1)])

    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        if not circuits:
            return []

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import EstimatorV2

        pass_manager = generate_preset_pass_manager(backend=self._backend, optimization_level=3)

        pubs = []
        for qc in circuits:
            isa_circuit = pass_manager.run(qc)
            isa_observable = self._observable.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_observable))

        estimator = EstimatorV2(mode=self._backend, options={"default_shots": self._n_shots})
        job = estimator.run(pubs)  # single batched job for the whole call
        result = job.result()
        return [float(np.real(pub_result.data.evs)) for pub_result in result]


# ---------------------------------------------------------------------------
# SpinQ NMR
# ---------------------------------------------------------------------------
class SpinQBackend(QuantumBackend):
    """Wraps the existing ``spinq_backend.SpinQNMRBackend``.

    ``SpinQNMRBackend.run`` already accepts a list of circuits (see
    ``spinq_backend.py``); this adapter simply passes the *whole* batch
    through in one call instead of looping circuit-by-circuit, and
    extracts each circuit's counts via ``result.get_counts(i)``.
    """

    def __init__(
        self,
        backend: Any,
        n_shots: int = DEFAULT_SHOTS,
        endianness: str = "big",
    ) -> None:
        if backend is None:
            raise ValueError("SpinQBackend requires a configured SpinQNMRBackend instance.")
        self._backend = backend
        self._n_shots = n_shots
        self._endianness = endianness

    def expectations(self, circuits: list[QuantumCircuit]) -> list[float]:
        if not circuits:
            return []

        measured = [qc.copy() for qc in circuits]
        for qc in measured:
            qc.measure_all()

        result = self._backend.run(measured, shots=self._n_shots)
        return [
            expectation_z_qubit0_from_counts(result.get_counts(i), self._endianness)
            for i in range(len(measured))
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_backend(
    backend_mode: str,
    backend: Any = None,
    n_shots: int = DEFAULT_SHOTS,
) -> QuantumBackend:
    """Resolve a concrete :class:`QuantumBackend` from a ``backend_mode`` string.

    This is the single place in the codebase that maps backend-mode
    strings to concrete implementations; ``VQC`` itself never inspects
    ``backend_mode`` once a :class:`QuantumBackend` has been constructed.

    Parameters
    ----------
    backend_mode : str
        One of ``"statevector"``, ``"aer_simulator"``, ``"ibm_simulator"``,
        ``"ibm_hardware"``, ``"spinq_nmr"`` (or any other string, which
        falls back to the generic :class:`AerBackend` counts-based path,
        matching the previous ``models.VQC._expectation`` fallback
        behaviour).
    backend : optional
        Pre-configured backend instance (``None`` for ``"statevector"``).
    n_shots : int
        Measurement shots for non-statevector backends.

    Returns
    -------
    QuantumBackend
    """
    endianness = BACKEND_ENDIANNESS.get(backend_mode, DEFAULT_ENDIANNESS)

    if backend_mode == "statevector":
        return StatevectorBackend()

    if backend_mode == "spinq_nmr":
        return SpinQBackend(backend, n_shots=n_shots, endianness=endianness)

    if backend_mode.startswith("ibm"):
        return IBMBackend(backend, n_shots=n_shots)

    # Aer simulator and any other legacy Qiskit-style backend.
    return AerBackend(backend, n_shots=n_shots, endianness=endianness)
