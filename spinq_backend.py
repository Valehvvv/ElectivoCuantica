"""SpinQ NMR backend integration.

Provides a backend wrapper that translates Qiskit circuits to SpinQ's
native format and executes them on the SpinQ 2-qubit NMR quantum computer.

Requirements
------------
- ``spinqit`` package installed from PyPI (``uv pip install spinqit``) in
  a dedicated Python 3.9 virtual environment created via
  ``scripts/setup_spinq.sh`` -- ``spinqit`` only publishes wheels for
  cp38/cp39/cp310, not 3.11+
- Python 3.9 recommended (the version used by the SpinQ tutorial); newer
  versions such as 3.12 may also work and are only a warning
- Access to the SpinQ NMR device via local network IP
"""

from __future__ import annotations

import importlib.util
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Environment guard (Fase 5)
# ---------------------------------------------------------------------------
# The main project environment targets Python >=3.12 (see ``pyproject.toml``
# / ``.python-version``). ``spinqit`` (the SpinQ NMR SDK) is installed from
# PyPI (``uv pip install spinqit``) in a dedicated venv 3.9; PyPI only
# publishes wheels for cp38/cp39/cp310, not 3.11+. It typically lives in a
# SEPARATE virtual environment -- see ``docs/spinq_setup.md`` -- but may
# also work on 3.12 (version mismatch is only warned about, not fatal).
#
# This check is intentionally cheap (``sys.version_info`` +
# ``importlib.util.find_spec``, which does NOT import ``spinqit``) so that
# importing this module in the main 3.12 environment stays side-effect
# free; the real ``import spinqit`` remains lazy inside ``connect()`` as
# before.
_REQUIRED_PYTHON = (3, 9)

_SPINQ_ENV_HELP = (
    "spinq_nmr backend requires a DEDICATED Python 3.9 environment with "
    "'spinqit' installed from PyPI -- it is NOT part of this project's "
    "main (Python >=3.12) environment/lockfile ('spinqit' only publishes "
    "wheels for cp38/cp39/cp310).\n"
    "\n"
    "To set it up:\n"
    "  1. Run scripts/setup_spinq.sh (creates .venv-spinq via "
    "'uv venv --python 3.9 .venv-spinq' and 'uv pip install spinqit').\n"
    "  2. Run scripts/run_spinq.sh to launch main.py with "
    "BACKEND_MODE=spinq_nmr inside that environment.\n"
    "\n"
    "Full reproducible instructions: docs/spinq_setup.md"
)


class SpinQEnvironmentError(RuntimeError):
    """Raised when ``spinq_nmr`` mode is requested but ``spinqit`` is not
    importable in the current environment.

    A Python version other than 3.9 is only a warning (see
    ``check_spinq_environment``), not a cause for this error.

    See ``docs/spinq_setup.md`` for how to create that environment.
    """


def check_spinq_environment() -> None:
    """Verify the current interpreter can run the SpinQ NMR backend.

    Checks (without importing ``spinqit``, so this stays cheap/safe to
    call from the main 3.12 environment):

    1. ``spinqit`` is importable (present on ``sys.path``). This is the
       only HARD requirement.
    2. The running Python is 3.9.x. This is the version used by the
       professor's tutorial, but NOT a hard requirement -- ``spinqit``
       may work on 3.12, so a mismatch is only a WARNING.

    Raises
    ------
    SpinQEnvironmentError
        With an actionable message pointing to ``docs/spinq_setup.md`` if
        ``spinqit`` is not importable.
    """
    if importlib.util.find_spec("spinqit") is None:
        problem = "'spinqit' is not importable in this environment."
        raise SpinQEnvironmentError(
            f"  - {problem}\n\n{_SPINQ_ENV_HELP}"
        )

    if sys.version_info[:2] != _REQUIRED_PYTHON:
        running = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"[spinq] aviso: probado en Python "
            f"{_REQUIRED_PYTHON[0]}.{_REQUIRED_PYTHON[1]} segun el tutorial; "
            f"intentando en Python {running}."
        )


class SpinQNMRBackend:
    """Backend that wraps a SpinQ 2-qubit NMR quantum computer.

    Parameters
    ----------
    ip : str
        IP address of the NMR device.
    port : int
        Port the device listens on (default 8989).
    username : str
        Authentication username.
    password : str
        Authentication password.
    task_name : str
        Identifier for the job on the device.
    shots : int
        Number of measurement shots (default 1024).
    """

    def __init__(
        self,
        ip: str,
        port: int = 8989,
        username: str = "",
        password: str = "",
        task_name: str = "VQC-Experiment",
        shots: int = 1024,
    ) -> None:
        self._ip = ip
        self._port = port
        self._username = username
        self._password = password
        self._task_name = task_name
        self._shots = shots
        self._engine = None
        self._compiler = None

    def connect(self) -> None:
        """Initialise the SpinQ engine and compiler.

        Raises
        ------
        SpinQEnvironmentError
            If ``spinqit`` is not importable in the current environment
            (see ``docs/spinq_setup.md``). Checked *before* attempting the
            (lazy) ``import spinqit`` so the error message is actionable
            rather than a bare ``ImportError``. A Python version other
            than 3.9 only prints a warning and proceeds.
        """
        check_spinq_environment()
        try:
            from spinqit import get_nmr, get_compiler

            self._engine = get_nmr()
            self._compiler = get_compiler("native")
            print(f"[spinq] Connected to NMR engine.")
        except ImportError:
            raise ImportError(
                "spinqit is not installed. "
                "Instalalo con: scripts/setup_spinq.sh "
                "(uv pip install spinqit en venv Python 3.9)."
            )

    def _qiskit_to_spinq(self, qc: Any) -> Any:
        """Translate a Qiskit ``QuantumCircuit`` to a SpinQ ``Circuit``.

        ``spinqit``'s gate objects (``Rx``, ``Ry``, ``Rz``, ``H``, ``CX``,
        ...) are plain ``spinqit.model.basic_gate.Gate`` instances, *not*
        callables/constructors. A gate (with its rotation angle, if any) is
        appended to a ``Circuit`` via the ``<<`` operator using the form
        ``circ << (gate, qubit_list, params)`` (``params`` omitted for
        gates that take none). This mirrors the pattern used by
        ``spinqit``'s own built-in translator
        (``spinqit.compiler.translator.qiskit_to_spinq``).

        Mapping of gates
        ----------------
        ==============  ==============================================
        Qiskit gate      SpinQ equivalent
        ==============  ==============================================
        ``ry(θ, q)``     ``circ << (Ry, [q], [θ])``
        ``rx(θ, q)``     ``circ << (Rx, [q], [θ])``
        ``rz(θ, q)``     ``circ << (Rz, [q], [θ])``
        ``cx(c, t)``     ``circ << (CX, [c, t])``
        ``ecr(c, t)``    decomposed via H + CX + H (approximation)
        ==============  ==============================================
        """
        from spinqit import CX, Circuit, H, Rx, Ry, Rz

        # SpinQ only supports 2 qubits on real NMR hardware
        circ = Circuit()
        qubits = circ.allocateQubits(2)

        # Iterate over Qiskit circuit instructions
        for instruction in qc.data:
            op = instruction.operation
            name = op.name
            qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]

            if name == "ry":
                theta = float(op.params[0])
                qidx = qubit_indices[0]
                circ << (Ry, [qubits[qidx]], [theta])

            elif name == "rx":
                theta = float(op.params[0])
                qidx = qubit_indices[0]
                circ << (Rx, [qubits[qidx]], [theta])

            elif name == "rz":
                theta = float(op.params[0])
                qidx = qubit_indices[0]
                circ << (Rz, [qubits[qidx]], [theta])

            elif name == "cx":
                ctrl, tgt = qubit_indices[0], qubit_indices[1]
                circ << (CX, [qubits[ctrl], qubits[tgt]])

            elif name == "ecr":
                ctrl, tgt = qubit_indices[0], qubit_indices[1]
                # ECR(c,t) ≈ (I⊗H) · CNOT · (I⊗H) ... approximate
                circ << (H, [qubits[tgt]])
                circ << (CX, [qubits[ctrl], qubits[tgt]])
                circ << (H, [qubits[tgt]])

            else:
                print(f"[spinq] Warning: gate '{name}' skipped (not supported).")

        return circ

    def run(self, circuits: list[Any], shots: int | None = None) -> Any:
        """Execute a batch of Qiskit circuits on the SpinQ NMR.

        Parameters
        ----------
        circuits : list[QuantumCircuit]
            Qiskit circuits to execute.  Each circuit *must* include
            measurements (``measure_all()``).
        shots : int, optional
            Override the default shot count.

        Returns
        -------
        SpinQJob
            An object whose ``result()`` method returns a ``SpinQResult``
            with a ``get_counts()`` method.
        """
        if self._engine is None or self._compiler is None:
            self.connect()

        from spinqit import NMRConfig

        results = _SpinQResultBatch()

        config = NMRConfig()
        config.configure_shots(shots if shots is not None else self._shots)
        config.configure_ip(self._ip)
        config.configure_port(self._port)
        config.configure_account(self._username, self._password)
        config.configure_task(self._task_name, self._task_name)

        for idx, qc in enumerate(circuits):
            spinq_circ = self._qiskit_to_spinq(qc)
            exe = self._compiler.compile(spinq_circ, 0)  # type: ignore[union-attr]
            raw = self._engine.execute(exe, config)  # type: ignore[union-attr]

            probs = raw.probabilities  # dict[str, float]  # type: ignore[union-attr]
            counts = _probabilities_to_counts(probs, shots if shots is not None else self._shots)
            results.add(counts)

            if (idx + 1) % 10 == 0:
                print(f"[spinq] Progress: {idx + 1}/{len(circuits)} circuits done.")

        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _probabilities_to_counts(
    probabilities: dict[str, float],
    shots: int,
) -> dict[str, int]:
    """Convert spinqit's probability dict to a Qiskit-style counts dict.

    SpinQ's NMR backend result exposes ``.probabilities`` as a **dict**
    ``{bitstring: prob}`` (see
    ``spinqit/backend/nmr_backend.py``, which iterates it via
    ``for k, v in res.probabilities.items(): idx = int(k, 2)``) - *not* an
    ordered ``[p00, p01, p10, p11]`` list. Each key is normalized to a
    2-bit string (``format(int(bitstring, 2), "02b")``) for consistency,
    matching spinqit's own ``int(k, 2)`` handling.

    The bit order within each key is whatever spinqit itself emits and is
    assumed to be ``p_{q0 q1}`` (i.e. ``bitstring[0]`` == qubit 0,
    "big-endian" - the *opposite* of Qiskit's own little-endian
    ``get_counts()`` convention). This assumption is declared explicitly
    via ``config.BACKEND_ENDIANNESS["spinq_nmr"] = "big"`` and consumed by
    ``observable.expectation_z_qubit0_from_counts`` in
    ``models.VQC._expectation`` - it is NOT hard-coded here.

    TODO(hardware-verification): this ordering has not been confirmed
    against real SpinQ NMR hardware output. If it turns out to be
    little-endian instead, update ``BACKEND_ENDIANNESS["spinq_nmr"]`` in
    ``config.py`` to ``"little"`` - no other code changes should be
    required. See ``docs/observable_convention.md``.
    """
    counts: dict[str, int] = {}
    for bitstring, p in probabilities.items():
        # normalizar a 2 bits, consistente con el int(k, 2) de spinqit
        key = format(int(bitstring, 2), "02b")
        count = int(round(float(p) * shots))
        if count > 0:
            counts[key] = counts.get(key, 0) + count
    return counts


class _SpinQResultBatch:
    """Minimal batch result mimicking Qiskit's job.result() interface."""

    def __init__(self) -> None:
        self._counts_list: list[dict[str, int]] = []

    def add(self, counts: dict[str, int]) -> None:
        self._counts_list.append(counts)

    def get_counts(self, circuit_index: int = 0) -> dict[str, int]:
        return self._counts_list[circuit_index]


def create_spinq_backend(
    ip: str,
    port: int = 8989,
    username: str = "",
    password: str = "",
    task_name: str = "VQC-Experiment",
    shots: int = 1024,
) -> SpinQNMRBackend:
    """Factory for a configured SpinQ NMR backend.

    Parameters
    ----------
    ip : str
        NMR device IP (provided by the lab).
    port : int
        Device port.
    username : str
        Auth username.
    password : str
        Auth password.
    task_name : str
        Task label visible on the device.
    shots : int
        Measurement shots.

    Returns
    -------
    SpinQNMRBackend
    """
    backend = SpinQNMRBackend(
        ip=ip,
        port=port,
        username=username,
        password=password,
        task_name=task_name,
        shots=shots,
    )
    backend.connect()
    return backend
