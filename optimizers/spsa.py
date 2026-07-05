"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

Reference:
    Spall, J. C. (1998). An overview of the simultaneous perturbation
    method for efficient optimization. Johns Hopkins APL Technical Digest.

Design
------
- Gradient-free stochastic optimization.
- Perturbation and step size decay over iterations.
- Returns result compatible with scipy.optimize.OptimizeResult interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
@dataclass
class SPSAResult:
    """Mimics scipy.optimize.OptimizeResult."""
    x: np.ndarray
    fun: float
    success: bool
    n_iter: int
    message: str
    fun_history: list[float] = field(default_factory=list)
def minimize_spsa(
    cost_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    a: float = 0.5,
    c: float = 0.5,
    alpha: float = 0.602,
    gamma: float = 0.101,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = 42,
    callback: Callable[[np.ndarray], None] | None = None,
) -> SPSAResult:
    """Minimize ``cost_fn`` using SPSA.

    Parameters
    ----------
    cost_fn : callable
        Function ``f(theta: np.ndarray) -> float``.
    x0 : np.ndarray
        Initial parameter vector.
    a : float
        Initial step size (decays as ``a_k = a / (k + 1 + A)^alpha``).
    c : float
        Initial perturbation size (decays as ``c_k = c / (k + 1)^gamma``).
    A : float
        Stability constant. Defaults to 10% of max_iter if not provided.
    alpha, gamma : float
        Decay exponents. Standard values: alpha=0.602, gamma=0.101.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on cost change.
    seed : int or None
        Random seed for reproducibility.
    callback : callable or None
        Called after each iteration with current theta.

    Returns
    -------
    SPSAResult with fields: x (best theta), fun (best cost), success,
    n_iter, message, fun_history.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(x0, dtype=np.float64).copy()
    n_params = theta.shape[0]
    A_val = 0.1 * max_iter  # stability constant

    best_theta = theta.copy()
    best_cost = cost_fn(theta)
    cost_prev = best_cost
    fun_history = [best_cost]

    for k in range(max_iter):
        ak = a / ((k + 1 + A_val) ** alpha)
        ck = c / ((k + 1) ** gamma)

        # Generate random perturbation ±1 Bernoulli
        delta = rng.choice([-1, 1], size=n_params).astype(np.float64)

        # Two-sided gradient estimate
        cost_plus = cost_fn(theta + ck * delta)
        cost_minus = cost_fn(theta - ck * delta)
        g_estimate = (cost_plus - cost_minus) / (2.0 * ck * delta)

        # Update
        theta -= ak * g_estimate
        current_cost = cost_fn(theta)

        fun_history.append(current_cost)

        # Track best
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = theta.copy()

        # Callback
        if callback is not None:
            callback(theta)

        # Convergence check
        if abs(current_cost - cost_prev) < tol:
            return SPSAResult(
                x=best_theta,
                fun=best_cost,
                success=True,
                n_iter=k + 1,
                message=f"Converged: |Δcost| < {tol} at iter {k + 1}",
                fun_history=fun_history,
            )

        cost_prev = current_cost

    return SPSAResult(
        x=best_theta,
        fun=best_cost,
        success=True,
        n_iter=max_iter,
        message=f"Reached max_iter={max_iter}",
        fun_history=fun_history,
    )