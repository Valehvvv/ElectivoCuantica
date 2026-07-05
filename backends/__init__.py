"""Backends module: new contract + legacy re-exports."""

from .legacy import (
    AerBackend,
    IBMBackend,
    SpinQBackend,
    StatevectorBackend,
    get_backend,
    QuantumBackend,
)

# Re-export contract classes for new code
from .contract import BackendError, StatevectorBackend as ContractStatevectorBackend

__all__ = [
    "QuantumBackend",
    "BackendError",
    "StatevectorBackend",
    "AerBackend",
    "IBMBackend",
    "SpinQBackend",
    "get_backend",
    "ContractStatevectorBackend",
]