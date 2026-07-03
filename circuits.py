"""Variational quantum circuit (ansatz) definitions for 2-qubit VQCs.

Each function builds a ``QuantumCircuit`` that encodes classical data via
Ry rotation gates and then applies a trainable ansatz.

Exported functions
------------------
- ``create_base_ansatz``   : 6 params, 2 CX gates, 2 entanglement layers
- ``create_reduced_ansatz``: 4 params, 1 CX gate, 1 entanglement layer
- ``create_hea_ansatz``    : 4 params, 1 ECR gate, HEA-style layout
- ``ansatz_circuit_info``  : extract depth, gate count, 2-qubit gate count
- ``draw_circuit``         : render a ``QuantumCircuit`` with matplotlib

The encoding is always::

    qc.ry(x0, q0) ; qc.ry(x1, q1)
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit, transpile


# ---------------------------------------------------------------------------
# 2-feature → angle encoding (shared by all ansätze)
# ---------------------------------------------------------------------------
def _encode(qc: QuantumCircuit, x: tuple[float, float] | list[float]) -> None:
    """Encode two classical features as Ry rotation angles."""
    qc.ry(float(x[0]), 0)
    qc.ry(float(x[1]), 1)


# ---------------------------------------------------------------------------
# Ansatz builders
# ---------------------------------------------------------------------------
def create_base_ansatz(
    x: tuple[float, float] | list[float],
    theta: tuple[float, ...] | list[float],
) -> QuantumCircuit:
    """Base ansatz (6 parameters).

    Structure (after angle encoding):
        Ry-θ0, Ry-θ1 → CX(0,1) → Ry-θ2, Ry-θ3 → CX(0,1) → Ry-θ4, Ry-θ5

    Parameters
    ----------
    x : array-like of length 2
        Classical input features.
    theta : array-like of length 6
        Variational parameters.

    Returns
    -------
    QuantumCircuit
        Complete variational circuit ready for measurement.
    """
    qc = QuantumCircuit(2)
    _encode(qc, x)

    # Layer 1
    qc.ry(float(theta[0]), 0)
    qc.ry(float(theta[1]), 1)
    qc.cx(0, 1)

    # Layer 2
    qc.ry(float(theta[2]), 0)
    qc.ry(float(theta[3]), 1)
    qc.cx(0, 1)

    # Layer 3
    qc.ry(float(theta[4]), 0)
    qc.ry(float(theta[5]), 1)

    return qc


def create_reduced_ansatz(
    x: tuple[float, float] | list[float],
    theta: tuple[float, ...] | list[float],
) -> QuantumCircuit:
    """Reduced ansatz (4 parameters, one entanglement layer removed).

    Structure (after angle encoding):
        Ry-θ0, Ry-θ1 → CX(0,1) → Ry-θ2, Ry-θ3

    Parameters
    ----------
    x : array-like of length 2
        Classical input features.
    theta : array-like of length 4
        Variational parameters.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(2)
    _encode(qc, x)

    # Layer 1
    qc.ry(float(theta[0]), 0)
    qc.ry(float(theta[1]), 1)
    qc.cx(0, 1)

    # Layer 2
    qc.ry(float(theta[2]), 0)
    qc.ry(float(theta[3]), 1)

    return qc


def create_hea_ansatz(
    x: tuple[float, float] | list[float],
    theta: tuple[float, ...] | list[float],
) -> QuantumCircuit:
    """Hardware-Efficient Ansatz (4 parameters, 1 ECR gate).

    Structure (after angle encoding):
        RX-θ0(q0), RZ-θ1(q0), RX-θ2(q1), RZ-θ3(q1) → ECR(0,1)

    Parameters
    ----------
    x : array-like of length 2
        Classical input features.
    theta : array-like of length 4
        Variational parameters.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(2)
    _encode(qc, x)

    # Local rotations
    qc.rx(float(theta[0]), 0)
    qc.rz(float(theta[1]), 0)
    qc.rx(float(theta[2]), 1)
    qc.rz(float(theta[3]), 1)

    # Entangling gate
    qc.ecr(0, 1)

    return qc


# ---------------------------------------------------------------------------
# Circuit metadata
# ---------------------------------------------------------------------------
def ansatz_circuit_info(
    qc: QuantumCircuit,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Return structural metrics for a quantum circuit after transpilation.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to analyse.
    optimization_level : int
        Transpilation optimisation level (0-3).

    Returns
    -------
    dict with keys ``depth``, ``size``, ``two_qubit_gates``, ``gate_counts``.
    """
    compiled = transpile(qc, optimization_level=optimization_level)
    ops = compiled.count_ops()
    two_qubit = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)

    return {
        "depth": compiled.depth(),
        "size": compiled.size(),
        "two_qubit_gates": two_qubit,
        "gate_counts": dict(ops),
    }


def draw_circuit(
    qc: QuantumCircuit,
    output: str = "mpl",
    scale: float = 0.7,
) -> Any:
    """Render a quantum circuit.

    Parameters
    ----------
    qc : QuantumCircuit
    output : str
        ``"mpl"`` (matplotlib), ``"text"``, or ``"latex"``.
    scale : float
        Scale factor for the drawing.

    Returns
    -------
    Matplotlib figure or text representation.
    """
    return qc.draw(output, scale=scale)
