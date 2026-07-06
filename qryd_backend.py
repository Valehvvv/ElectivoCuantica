"""QRydDemo backend integration.

Provides utilities to authenticate, select backends, and execute circuits
on the QRydDemo platform (trapped Rydberg atom quantum computing).

Uses ``qiskit_qryd_provider`` package when available, with graceful fallback.

Exported functions
------------------
- ``get_qryd_provider`` : obtain a configured QRydDemo provider
- ``select_qryd_backend`` : select a backend from the provider
- ``create_qryd_backend`` : factory that connects and returns a backend
"""

from __future__ import annotations

from typing import Any


def get_qryd_provider() -> Any | None:
    """Return a QRydDemo provider instance.

    Requires ``qiskit_qryd_provider`` package and a valid ``QRYD_API_TOKEN``.

    Returns
    -------
    Provider instance or ``None`` if unavailable.
    """
    import os

    token = os.environ.get("QRYD_API_TOKEN", "")
    if not token:
        print("[qryd] QRYD_API_TOKEN is not set. Add it to .env")
        return None

    try:
        from qiskit_qryd_provider import QRydProvider

        provider = QRydProvider(token)
        print("[qryd] Connected via QRydProvider.")
        return provider
    except ImportError:
        print(
            "[qryd] qiskit-qryd-provider not installed. "
            "Install with: pip install qiskit-qryd-provider"
        )
        return None
    except Exception as exc:
        print(f"[qryd] Could not connect: {exc}")
        return None


def list_available_backends(provider: Any) -> list[str]:
    """Return a list of backend names available via this provider."""
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
        Explicit backend name (e.g. ``"qryd_emulator$square"``).
        If ``None``, uses ``QRYD_BACKEND_NAME`` from config.

    Returns
    -------
    Backend instance or ``None``.
    """
    if provider is None:
        return None

    import os

    if backend_name is None:
        backend_name = os.environ.get("QRYD_BACKEND_NAME", "qryd_emulator$square")

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


def create_qryd_backend(
    backend_name: str | None = None,
) -> Any | None:
    """Factory: connect to QRydDemo and return a configured backend.

    Parameters
    ----------
    backend_name : str, optional
        Explicit backend name (e.g. ``"qryd_emulator$square"``).

    Returns
    -------
    Backend instance or ``None``.
    """
    provider = get_qryd_provider()
    if provider is None:
        return None
    return select_qryd_backend(provider, backend_name=backend_name)
