"""IBM Quantum Runtime integration module.

Provides utilities to authenticate, select backends, transpile circuits,
and execute jobs on IBM Quantum hardware or cloud simulators.

Uses the modern Qiskit Runtime primitives where available, with graceful
fallback when the package is not installed.

Exported functions
------------------
- ``setup_ibm_backend`` : obtain a configured IBM backend
- ``transpile_for_backend`` : transpile a circuit for a specific backend
- ``run_on_backend`` : submit and retrieve job results
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit, transpile

from config import IBMQ_BACKEND_HARDWARE, IBMQ_BACKEND_SIMULATOR, IBMQ_INSTANCE, IBMQ_TOKEN


def get_ibm_runtime_service() -> Any:
    """Return a Qiskit Runtime ``Session`` (or fallback ``AccountProvider``).

    If ``qiskit_ibm_runtime`` is installed it is used; otherwise
    ``qiskit.providers.ibmq`` is attempted.

    Returns
    -------
    Service object or ``None`` if no credentials are available.
    """
    if not IBMQ_TOKEN:
        print(
            "[ibm_runtime] IBMQ_TOKEN is not set. "
            "Export it as an environment variable or set it in config.py."
        )
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=IBMQ_TOKEN,
            instance=IBMQ_INSTANCE,
        )
        print("[ibm_runtime] Connected via QiskitRuntimeService.")
        return service
    except ImportError:
        pass

    try:
        from qiskit import IBMQ

        IBMQ.save_account(IBMQ_TOKEN, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=IBMQ_INSTANCE)
        print("[ibm_runtime] Connected via legacy IBMQ provider.")
        return provider
    except ImportError:
        print(
            "[ibm_runtime] Neither qiskit-ibm-runtime nor qiskit "
            "IBMQ provider is available."
        )
        return None


def select_backend(
    service: Any,
    backend_name: str | None = None,
    simulator: bool = False,
) -> Any | None:
    """Select a specific backend from an IBM service.

    Parameters
    ----------
    service : QiskitRuntimeService or IBMQ provider.
    backend_name : str, optional
        Explicit backend name.  If ``None``, one is chosen based on
        ``simulator``.
    simulator : bool
        If ``True`` and ``backend_name`` is ``None``, pick the default IBM
        cloud simulator.

    Returns
    -------
    Backend instance or ``None``.
    """
    if service is None:
        return None

    if backend_name is None:
        backend_name = IBMQ_BACKEND_SIMULATOR if simulator else IBMQ_BACKEND_HARDWARE

    try:
        backend = service.backend(backend_name)
        print(f"[ibm_runtime] Selected backend: {backend.name}")
        return backend
    except Exception as exc:
        print(f"[ibm_runtime] Could not select backend '{backend_name}': {exc}")
        return None


def transpile_for_backend(
    qc: QuantumCircuit,
    backend: Any,
    optimization_level: int = 3,
) -> QuantumCircuit:
    """Transpile a circuit for a specific backend.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to transpile.
    backend : Backend instance.
    optimization_level : int
        Transpiler optimisation level.

    Returns
    -------
    QuantumCircuit
    """
    try:
        tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
    except Exception:
        # Fallback: transpile without backend-specific coupling map
        tqc = transpile(qc, optimization_level=optimization_level)

    return tqc


def run_on_backend(
    qc: QuantumCircuit,
    backend: Any,
    shots: int = 1024,
) -> dict[str, int]:
    """Run a circuit on an IBM backend and return measurement counts.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit **with measurements**.
    backend : Backend instance.
    shots : int
        Number of shots.

    Returns
    -------
    dict[str, int] counts.
    """
    qc = transpile_for_backend(qc, backend)
    job = backend.run(qc, shots=shots)
    result = job.result()
    return result.get_counts(qc)
