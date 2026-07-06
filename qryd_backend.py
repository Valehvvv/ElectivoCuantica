"""QRydDemo backend integration.

Provides utilities to authenticate, select backends, and execute circuits
on the QRydDemo platform (trapped Rydberg atom quantum computing).

Uses ``qiskit_qryd`` provider when available, with graceful fallback.

Exported functions
------------------
- ``get_qryd_provider`` : obtain a configured QRydDemo provider
- ``select_qryd_backend`` : select a backend from the provider
- ``run_on_qryd`` : submit and retrieve job results
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit, transpile

from config import QRYD_BACKEND_NAME, QRYD_TOKEN


def get_qryd_provider() -> Any | None:
    """Return a QRydDemo provider instance.

    Requires ``qiskit_qryd`` package and a valid ``QRYD_TOKEN``.

    Returns
    -------
    Provider instance or ``None`` if unavailable.
    """
    if not QRYD_TOKEN:
        print("[qryd] QRYD_TOKEN is not set. Add it to .env")
        return None

    try:
        from qiskit_qryd import QRydProvider

        provider = QRydProvider(token=QRYD_TOKEN)
        print("[qryd] Connected via QRydProvider.")
        return provider
    except ImportError:
        print(
            "[qryd] qiskit-qryd not installed. Install with: pip install qiskit-qryd"
        )
        return None
    except Exception as exc:
        print(f"[qryd] Could not connect: {exc}")
        return None


def list_available_backends(provider: Any) -> list[str]:
    """Return a list of backend names available via this provider.

    Parameters
    ----------
    provider : QRydProvider instance.

    Returns
    -------
    list[str]
    """
    if provider is None:
        return []
    try:
        return [b.name for b in provider.backends()]
    except Exception:
        return []


def select_qryd_backend(
    provider: Any,
    backend_name: str | None = None,
) -> Any | None:
    """Select a specific backend from a QRydDemo provider.

    Parameters
    ----------
    provider : QRydProvider instance.
    backend_name : str, optional
        Explicit backend name. If ``None``, uses ``QRYD_BACKEND_NAME``
        from config.

    Returns
    -------
    Backend instance or ``None``.
    """
    if provider is None:
        return None

    if backend_name is None:
        backend_name = QRYD_BACKEND_NAME

    try:
        backend = provider.get_backend(backend_name)
        print(f"[qryd] Selected backend: {backend.name}")
        return backend
    except Exception:
        all_backends = provider.backends()
        if not all_backends:
            print("[qryd] No backends available.")
            return None

        print(f"[qryd] Available backends: {[b.name for b in all_backends]}")
        backend = all_backends[0]
        print(f"[qryd] Auto-selected: {backend.name}")
        return backend


def transpile_for_qryd(
    qc: QuantumCircuit,
    backend: Any,
    optimization_level: int = 3,
) -> QuantumCircuit:
    """Transpile a circuit for a specific QRydDemo backend.

    Parameters
    ----------
    qc : QuantumCircuit
    backend : Backend instance.
    optimization_level : int

    Returns
    -------
    QuantumCircuit
    """
    try:
        tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
    except Exception:
        tqc = transpile(qc, optimization_level=optimization_level)
    return tqc


def run_on_qryd(
    qc: QuantumCircuit,
    backend: Any,
    shots: int = 1024,
) -> dict[str, int]:
    """Run a circuit on a QRydDemo backend and return counts.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit **with measurements**.
    backend : Backend instance.
    shots : int

    Returns
    -------
    dict[str, int] counts.
    """
    tqc = transpile_for_qryd(qc, backend)
    job = backend.run(tqc, shots=shots)
    result = job.result()
    return result.get_counts()


def create_qryd_backend(
    backend_name: str | None = None,
) -> Any | None:
    """Factory: connect to QRydDemo and return a configured backend.

    Parameters
    ----------
    backend_name : str, optional
        Explicit backend name.

    Returns
    -------
    Backend instance or ``None``.
    """
    provider = get_qryd_provider()
    if provider is None:
        return None
    return select_qryd_backend(provider, backend_name=backend_name)
