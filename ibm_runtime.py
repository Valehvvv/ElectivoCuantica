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

        kwargs: dict[str, str] = {
            "channel": "ibm_quantum_platform",
            "token": IBMQ_TOKEN,
        }
        if IBMQ_INSTANCE:
            kwargs["instance"] = IBMQ_INSTANCE

        service = QiskitRuntimeService(**kwargs)
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


def list_available_backends(service: Any) -> list[str]:
    """Return a list of backend names available via this service.

    Parameters
    ----------
    service : QiskitRuntimeService or IBMQ provider.

    Returns
    -------
    list[str]
    """
    if service is None:
        return []
    try:
        return [b.name for b in service.backends()]
    except Exception:
        return []


def select_backend(
    service: Any,
    backend_name: str | None = None,
    simulator: bool = False,
) -> Any | None:
    """Select a specific backend from an IBM service.

    If ``backend_name`` is ``None``, auto-select least-busy operational
    QPU (or first available simulator) depending on ``simulator``.

    Parameters
    ----------
    service : QiskitRuntimeService or IBMQ provider.
    backend_name : str, optional
        Explicit backend name.
    simulator : bool
        If ``True`` and ``backend_name`` is ``None``, pick a simulator.

    Returns
    -------
    Backend instance or ``None``.
    """
    if service is None:
        return None

    if backend_name is not None:
        try:
            backend = service.backend(backend_name)
            print(f"[ibm_runtime] Selected backend: {backend.name}")
            return backend
        except Exception:
            print(f"[ibm_runtime] Backend '{backend_name}' not found, auto-detecting...")

    all_backends = service.backends()
    if not all_backends:
        print("[ibm_runtime] No backends available.")
        return None

    print(f"[ibm_runtime] Available backends: {[b.name for b in all_backends]}")

    if simulator:
        sim_backends = [b for b in all_backends if b.simulator]
        if sim_backends:
            backend = sim_backends[0]
            print(f"[ibm_runtime] Auto-selected simulator: {backend.name}")
            return backend
        print("[ibm_runtime] No simulator found, using first available backend.")

    try:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
        print(f"[ibm_runtime] Auto-selected least-busy QPU: {backend.name}")
        return backend
    except Exception as exc:
        print(f"[ibm_runtime] least_busy failed ({exc!r}), falling back to first QPU.")
    real_backends = [b for b in all_backends if not b.simulator]
    if real_backends:
        backend = real_backends[0]
        print(f"[ibm_runtime] Auto-selected QPU: {backend.name}")
        return backend

    backend = all_backends[0]
    print(f"[ibm_runtime] Auto-selected backend: {backend.name}")
    return backend


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
