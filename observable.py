"""Single source of truth for the VQC readout observable.

See ``docs/observable_convention.md`` for the full design rationale.  In
short: this project's classifier reads out **circuit qubit index
``MEASURED_QUBIT_INDEX`` (currently 1)**, not qubit index 0, in order to
preserve the historical (pre-Phase-1) statevector behaviour captured in
``results/baseline_statevector.csv``.  Every backend (statevector, Aer,
IBM Runtime, SpinQ NMR) must agree on this *same physical qubit*, even
though each backend may report measurement outcomes using a different
bitstring ordering ("endianness").

Two bitstring orderings are supported:

- ``"little"`` (Qiskit convention): ``bitstring[0]`` is the *highest*
  qubit index (qubit ``n - 1``); ``bitstring[-1]`` is qubit 0.  This is
  how ``qiskit`` / ``qiskit-aer`` / ``qiskit-ibm-runtime`` all report
  ``get_counts()`` results.
- ``"big"``: ``bitstring[0]`` is qubit 0; ``bitstring[-1]`` is qubit
  ``n - 1``.  This is the assumed convention for SpinQ NMR counts as
  produced by ``spinq_backend._probabilities_to_counts`` (probability
  list ``[p00, p01, p10, p11]`` read as ``p_{q0 q1}``).  This has **not**
  been verified against real SpinQ hardware output - see the TODO in
  ``config.py`` / ``docs/observable_convention.md``.
"""

from __future__ import annotations

# Physical circuit qubit whose Pauli-Z expectation value is used as the
# classifier's readout.  Kept at index 1 (not the "natural" index 0) for
# historical/back-compat reasons - see docs/observable_convention.md.
MEASURED_QUBIT_INDEX: int = 1


def expectation_z_from_counts(
    counts: dict[str, int],
    qubit_index: int,
    endianness: str,
) -> float:
    """Compute ``<Z>`` on a given circuit qubit from a Qiskit-style counts dict.

    Parameters
    ----------
    counts : dict[str, int]
        Mapping of bitstring -> shot count, as returned by
        ``result.get_counts()`` (Qiskit) or an equivalent adapter for
        other backends (e.g. SpinQ).
    qubit_index : int
        Physical circuit qubit index (0-based) whose ``Z`` expectation is
        requested.
    endianness : {"little", "big"}
        Bitstring ordering convention:

        - ``"little"``: ``bitstring[0]`` == qubit ``n - 1`` (Qiskit).
        - ``"big"``:    ``bitstring[0]`` == qubit 0.

    Returns
    -------
    float
        ``<Z>`` in ``[-1, 1]``. Returns ``0.0`` if ``counts`` is empty.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    if endianness not in ("little", "big"):
        raise ValueError(
            f"Unknown endianness '{endianness}'; expected 'little' or 'big'."
        )

    exp_val = 0.0
    for bitstring, count in counts.items():
        n_bits = len(bitstring)
        if endianness == "little":
            pos = n_bits - 1 - qubit_index
        else:  # "big"
            pos = qubit_index
        bit = bitstring[pos]
        parity = 1 if bit == "0" else -1
        exp_val += parity * count

    return exp_val / total


def expectation_z_qubit0_from_counts(
    counts: dict[str, int],
    endianness: str,
) -> float:
    """Compute the VQC readout ``<Z>`` value from a counts dict.

    This is the single utility used by :meth:`models.VQC._expectation` for
    every counts-based backend (Aer, IBM, SpinQ NMR).  Despite the name
    (kept for API-contract consistency with the Phase 1 design spec), it
    targets ``observable.MEASURED_QUBIT_INDEX`` (currently physical qubit
    1), not necessarily circuit-wire index 0.  See
    ``docs/observable_convention.md`` for the full rationale.

    Parameters
    ----------
    counts : dict[str, int]
        Measurement counts, e.g. from ``result.get_counts()``.
    endianness : {"little", "big"}
        Bitstring ordering convention for this backend (see
        :func:`expectation_z_from_counts`).

    Returns
    -------
    float
        ``<Z>`` in ``[-1, 1]``.
    """
    return expectation_z_from_counts(counts, MEASURED_QUBIT_INDEX, endianness)
