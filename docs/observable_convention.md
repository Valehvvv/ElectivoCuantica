# Observable / readout qubit convention

## Problem

The VQC classifier maps a single Pauli-`Z` expectation value to a class
probability (`p = (⟨Z⟩ + 1) / 2`). Before Phase 1, the extraction of this
`⟨Z⟩` value was implemented independently in three places and was
**inconsistent**:

1. **Statevector path** (`models.VQC._expectation`): used
   `SparsePauliOp("ZI")`. In Qiskit's tensor-product convention the
   *leftmost* character of a Pauli string acts on the *highest*-indexed
   qubit. For 2 qubits, `"ZI"` therefore applies `Z` to **circuit qubit
   index 1** and `I` to qubit 0 - despite the code comment claiming
   "qubit 0".
2. **Counts path (Aer / IBM)**: used `parity = 1 if bitstring[0] == "0"
   else -1`. Qiskit's `get_counts()` bitstrings are also written
   MSB-first, i.e. `bitstring[0]` corresponds to the *highest*-indexed
   qubit (qubit 1 for a 2-qubit circuit). This happens to be **the same
   physical qubit** as the statevector path, so Aer/IBM and statevector
   were mutually consistent (both measuring qubit 1), even though the
   "qubit 0" comment was wrong for all three.
3. **SpinQ NMR path** (`spinq_backend._probabilities_to_counts`): builds
   counts from a raw probability list `[p00, p01, p10, p11]` mapped
   directly onto bitstrings `"00", "01", "10", "11"`. The natural
   reading of `p00, p01, p10, p11` is `p_{q0 q1}`, i.e.
   `bitstring[0]` == qubit 0 ("big-endian"), the **opposite** ordering
   from Qiskit's own `get_counts()`. If SpinQ counts were fed through the
   same `bitstring[0] == "0"` parity check used for Aer/IBM, the SpinQ
   path would silently measure a *different physical qubit* (qubit 0)
   than statevector/Aer/IBM (qubit 1).

We verified the statevector/counts equivalence empirically (see
`Verification` below): applying `X` to physical qubit 1 flips the Aer
counts to `"1x"` bitstrings (`bitstring[0] == "1"`) **and** flips the
`"ZI"` statevector expectation to `-1`, while applying `X` to physical
qubit 0 does neither. This confirms both paths already agreed on
measuring circuit qubit index **1**.

## Decision

Per the Phase 1 scope (minimize changes to the Phase 0 statevector
baseline in `results/baseline_statevector.csv`), we chose **option (b)**:

> Keep measuring physical circuit qubit **index 1** (not "true" qubit 0)
> in the statevector and Aer/IBM (Qiskit) paths - i.e. do **not** change
> `OBSERVABLE_PAULI` or the statevector code path at all - and instead
> make the SpinQ NMR path measure **the same physical qubit (index 1)**
> by correctly interpreting its bitstring ordering.

This was preferred over option (a) ("move everything to the real qubit
0") because option (a) would change `OBSERVABLE_PAULI` from `"ZI"` to
`"IZ"` (and flip the counts-parity bit position for Aer/IBM), which
would change the exact statevector expectation values computed for every
sample and therefore the Accuracy/F1 numbers captured in the Phase 0
baseline - i.e. it would *fail* the Phase 0 regression test by design,
not by bug. Option (b) requires **zero** change to the deterministic
statevector path, so the Phase 0 regression test
(`tests/test_regression_baseline.py`) continues to pass unchanged.

## Implementation

All expectation-value extraction from measurement counts is now
centralized in `observable.py`:

- `observable.MEASURED_QUBIT_INDEX = 1` - the single physical qubit
  index that is measured, for every backend.
- `observable.expectation_z_from_counts(counts, qubit_index, endianness)`
  - generic helper, given an explicit bitstring endianness.
- `observable.expectation_z_qubit0_from_counts(counts, endianness)` - the
  function actually called from `models.VQC._expectation` for every
  counts-based backend (Aer, IBM Runtime, SpinQ NMR). The name is kept
  for API-contract consistency with the Phase 1 design spec; it targets
  `MEASURED_QUBIT_INDEX` (currently 1), which is documented explicitly in
  its docstring to avoid confusion.

`config.BACKEND_ENDIANNESS` declares, per `backend_mode`, which bitstring
ordering that backend's `get_counts()`-equivalent output uses:

| `backend_mode`   | endianness | bitstring\[0\] means | verified? |
|-------------------|------------|------------------------|-----------|
| `statevector`     | `"little"` | n/a (analytic `SparsePauliOp`, not counts) | yes (Qiskit spec) |
| `aer_simulator`   | `"little"` | qubit 1 (highest index) | yes (Qiskit spec + empirical check below) |
| `ibm_simulator`   | `"little"` | qubit 1 (highest index) | yes (Qiskit spec) |
| `ibm_hardware`    | `"little"` | qubit 1 (highest index) | yes (Qiskit spec) |
| `spinq_nmr`       | `"big"`    | qubit 0 (lowest index)  | **NO - TODO, see below** |

The statevector path itself is untouched: it still uses
`config.OBSERVABLE_PAULI = "ZI"` directly via `SparsePauliOp`, which is
documented as being kept consistent with `MEASURED_QUBIT_INDEX = 1`.

## Verification performed

Empirically confirmed with Qiskit 2.5.0 / Qiskit Aer, via a scratch
script:

```python
qc = QuantumCircuit(2); qc.x(1)          # flip physical qubit 1
Statevector.from_instruction(qc).expectation_value(SparsePauliOp("ZI"))
# -> -1.0
AerSimulator().run(qc_with_measure_all, shots=100).result().get_counts()
# -> {'10': 100}   (bitstring[0] == '1')

qc2 = QuantumCircuit(2); qc2.x(0)        # flip physical qubit 0
Statevector.from_instruction(qc2).expectation_value(SparsePauliOp("ZI"))
# -> +1.0 (unchanged)
AerSimulator().run(qc2_with_measure_all, shots=100).result().get_counts()
# -> {'01': 100}   (bitstring[0] == '0', unchanged)
```

This confirms `"ZI"` (statevector) and `bitstring[0]` (Aer counts) both
target physical qubit 1, and that they are mutually consistent - exactly
matching the behaviour implemented for the "little" endianness case in
`observable.py`.

## Outstanding assumption (NOT hardware-verified)

The SpinQ NMR bitstring ordering (`"big"`, i.e. `bitstring[0]` == qubit
0) is an **assumption** based on the most natural reading of
`spinq_backend._probabilities_to_counts`'s `[p00, p01, p10, p11]` -> the
convention has **not** been verified against real SpinQ NMR hardware
output, because SpinQ hardware access / `spinqit` was not available in
this environment.

If/when SpinQ hardware access becomes available, verify by running a
circuit that applies `X` only to the qubit encoding `x[1]` (the second
Ry-encoded feature - the ansatz's "qubit 1") and checking which
character of the resulting SpinQ bitstring flips. If it is
`bitstring[0]`, the current `"big"` assumption is correct. If it is
`bitstring[-1]` (i.e. matches Qiskit's own "little" ordering instead),
update `config.BACKEND_ENDIANNESS["spinq_nmr"]` to `"little"` - no other
code changes are needed, since all extraction logic is centralized in
`observable.py`.
