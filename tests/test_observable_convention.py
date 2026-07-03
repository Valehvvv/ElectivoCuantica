"""Phase 1 tests: unified observable / readout-qubit convention.

Verifies that:

1. ``observable.expectation_z_qubit0_from_counts`` correctly extracts
   ``<Z>`` on ``observable.MEASURED_QUBIT_INDEX`` for both supported
   bitstring endianness conventions ("little" == Qiskit, "big" ==
   assumed SpinQ ordering), using known basis states.
2. The statevector path (``SparsePauliOp("ZI")`` via
   ``config.OBSERVABLE_PAULI``) agrees in *sign* with the counts path
   (via Aer simulation) for the same circuit, confirming both target the
   same physical qubit.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp

from config import BACKEND_ENDIANNESS, OBSERVABLE_PAULI
from observable import (
    MEASURED_QUBIT_INDEX,
    expectation_z_from_counts,
    expectation_z_qubit0_from_counts,
)


# ---------------------------------------------------------------------------
# Known-state tests: qiskit ("little") counts convention
# ---------------------------------------------------------------------------
def test_little_endian_qubit0_ground_state():
    """|q1 q0> = |00> -> Z on qubit 0 is +1 regardless of endianness."""
    counts = {"00": 1000}
    assert expectation_z_from_counts(counts, qubit_index=0, endianness="little") == pytest.approx(1.0)


def test_little_endian_flip_on_target_qubit():
    """Qiskit bitstring '10' means qubit1=1, qubit0=0 (bitstring[0]=qubit1).

    So <Z> on qubit 1 must be -1, and <Z> on qubit 0 must be +1.
    """
    counts = {"10": 1000}
    assert expectation_z_from_counts(counts, qubit_index=1, endianness="little") == pytest.approx(-1.0)
    assert expectation_z_from_counts(counts, qubit_index=0, endianness="little") == pytest.approx(1.0)


def test_little_endian_flip_on_qubit0():
    """Qiskit bitstring '01' means qubit1=0, qubit0=1.

    So <Z> on qubit 0 must be -1, and <Z> on qubit 1 must be +1.
    """
    counts = {"01": 1000}
    assert expectation_z_from_counts(counts, qubit_index=0, endianness="little") == pytest.approx(-1.0)
    assert expectation_z_from_counts(counts, qubit_index=1, endianness="little") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Known-state tests: assumed SpinQ ("big") counts convention
# ---------------------------------------------------------------------------
def test_big_endian_flip_on_qubit0():
    """SpinQ-style bitstring '10' assumed to mean qubit0=1, qubit1=0."""
    counts = {"10": 1000}
    assert expectation_z_from_counts(counts, qubit_index=0, endianness="big") == pytest.approx(-1.0)
    assert expectation_z_from_counts(counts, qubit_index=1, endianness="big") == pytest.approx(1.0)


def test_big_endian_flip_on_qubit1():
    """SpinQ-style bitstring '01' assumed to mean qubit0=0, qubit1=1."""
    counts = {"01": 1000}
    assert expectation_z_from_counts(counts, qubit_index=1, endianness="big") == pytest.approx(-1.0)
    assert expectation_z_from_counts(counts, qubit_index=0, endianness="big") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The project's readout function: |0>|1> vs |1>|0> for each convention
# ---------------------------------------------------------------------------
def test_qubit0_readout_state_01_little():
    """|q1=0, q0=1> (Qiskit '01') and |q1=1, q0=0> (Qiskit '10').

    MEASURED_QUBIT_INDEX is qubit 1 (see docs/observable_convention.md),
    so the readout should flip sign between these two states.
    """
    exp_q1_zero = expectation_z_qubit0_from_counts({"01": 1000}, endianness="little")
    exp_q1_one = expectation_z_qubit0_from_counts({"10": 1000}, endianness="little")
    assert exp_q1_zero == pytest.approx(1.0)
    assert exp_q1_one == pytest.approx(-1.0)


def test_qubit0_readout_state_01_big():
    """Same physical states under the assumed SpinQ ('big') convention.

    Physical qubit1=0 -> SpinQ bitstring '00' or '10' (qubit0 free);
    physical qubit1=1 -> SpinQ bitstring '01' or '11'.
    Fix qubit0=0 for both cases below.
    """
    exp_q1_zero = expectation_z_qubit0_from_counts({"00": 1000}, endianness="big")
    exp_q1_one = expectation_z_qubit0_from_counts({"01": 1000}, endianness="big")
    assert exp_q1_zero == pytest.approx(1.0)
    assert exp_q1_one == pytest.approx(-1.0)


def test_measured_qubit_index_is_documented_and_used():
    """Sanity check that the module-level constant matches config expectations."""
    assert MEASURED_QUBIT_INDEX == 1
    assert OBSERVABLE_PAULI == "ZI"
    assert BACKEND_ENDIANNESS["aer_simulator"] == "little"
    assert BACKEND_ENDIANNESS["spinq_nmr"] == "big"


# ---------------------------------------------------------------------------
# Cross-path consistency: statevector (SparsePauliOp) vs counts (Aer)
# ---------------------------------------------------------------------------
def _statevector_expectation(qc: QuantumCircuit) -> float:
    observable = SparsePauliOp.from_list([(OBSERVABLE_PAULI, 1)])
    state = Statevector.from_instruction(qc)
    return float(np.real(state.expectation_value(observable)))


def _aer_counts_expectation(qc: QuantumCircuit, shots: int = 2000) -> float:
    aer = pytest.importorskip("qiskit_aer")
    backend = aer.AerSimulator()

    qc_meas = qc.copy()
    qc_meas.measure_all()
    job = backend.run(qc_meas, shots=shots)
    counts = job.result().get_counts()

    endianness = BACKEND_ENDIANNESS["aer_simulator"]
    return expectation_z_qubit0_from_counts(counts, endianness)


@pytest.mark.parametrize(
    "flip_qubit,expected_sign",
    [
        (1, -1),  # flipping the measured qubit (index 1) should flip the sign
        (0, 1),  # flipping the other qubit should NOT flip the sign
    ],
)
def test_statevector_and_counts_agree_in_sign(flip_qubit, expected_sign):
    qc = QuantumCircuit(2)
    qc.x(flip_qubit)

    sv_exp = _statevector_expectation(qc)
    counts_exp = _aer_counts_expectation(qc)

    assert np.sign(sv_exp) == expected_sign
    assert np.sign(counts_exp) == expected_sign
    assert sv_exp == pytest.approx(counts_exp, abs=0.1)
