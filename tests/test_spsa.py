"""Tests for SPSA optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from optimizers.spsa import minimize_spsa
def test_spsa_converges_parabola() -> None:
    """Minimize f(x)=sum(x_i^2). Should converge near zero."""
    result = minimize_spsa(
        lambda t: float(np.sum(t ** 2)),
        np.array([2.0, -1.0]),
        max_iter=100,
        seed=42,
    )
    assert result.success
    assert result.fun < 0.5  # should be well below initial cost of 5
    assert result.n_iter > 0
    assert len(result.fun_history) > 1
def test_spsa_fun_history_length() -> None:
    result = minimize_spsa(
        lambda t: float(np.sum(t ** 2)),
        np.array([1.0]),
        max_iter=10,
        seed=42,
    )
    assert len(result.fun_history) == result.n_iter + 1  # initial + per iter
def test_spsa_reproducible_seed() -> None:
    r1 = minimize_spsa(lambda t: float(np.sum(t ** 2)), np.array([1.0]), max_iter=5, seed=42)
    r2 = minimize_spsa(lambda t: float(np.sum(t ** 2)), np.array([1.0]), max_iter=5, seed=42)
    np.testing.assert_array_almost_equal(r1.x, r2.x)
    assert r1.fun_history == r2.fun_history
def test_spsa_callback_called() -> None:
    calls = []

    def cb(xk: np.ndarray) -> None:
        calls.append(xk.copy())

    minimize_spsa(
        lambda t: float(np.sum(t ** 2)),
        np.array([1.0]),
        max_iter=5,
        seed=42,
        callback=cb,
    )
    assert len(calls) == 5  # one per iteration
def test_spsa_single_param() -> None:
    result = minimize_spsa(
        lambda t: float((t[0] - 3.0) ** 2),
        np.array([0.0]),
        max_iter=100,
        seed=42,
    )
    assert result.success
    assert abs(result.x[0] - 3.0) < 0.5
def test_spsa_with_tol() -> None:
    result = minimize_spsa(
        lambda t: float(np.sum(t ** 2)),
        np.array([2.0]),
        max_iter=1000,
        tol=1e-3,
        seed=42,
    )
    assert result.success
    assert "Converged" in result.message