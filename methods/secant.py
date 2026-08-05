from __future__ import annotations

from dataclasses import dataclass 
from typing import Callable, List

Number = float
Func = Callable[[Number], Number]

@dataclass(frozen=True)
class SecantResult:
    root: float
    iterations: int
    converged: bool
    history: List[float]


def secant_method(
    f: Func,
    x0: float,
    x1: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    min_denom: float = 1e-14,
) -> SecantResult:
    """
    Solve a scalar nonlinear equation using the secant method.

    Parameters
    ----------
    f : Callable[[float], float]
        Scalar function for which a root is sought.
    x0 : float
        First initial approximation.
    x1 : float
        Second initial approximation.
    tol : float, optional
        Absolute convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    min_denom : float, optional
    Minimum permitted absolute secant denominator. The solver stops
    without convergence when the denominator falls below this value.
    
    Returns
    -------
    SecantResult
        Structured result containing the computed root, convergence
        status, iteration count, and history of approximations.

    Notes
    -----
    The secant method is derivative-free and converges superlinearly
    under suitable conditions, although convergence is not guaranteed.
    """
    fx0 = f(x0)
    fx1 = f(x1)

    history: List[float] = [x0, x1]

    for k in range(1, max_iter + 1):
        denom = fx1 - fx0
        if abs(denom) < min_denom:
            return SecantResult(root=x1, iterations=k - 1, converged=False, history=history)

        x2 = x1 - fx1 * (x1 - x0) / denom
        history.append(x2)

        if abs(x2 - x1) <= tol:
            return SecantResult(root=x2, iterations=k, converged=True, history=history)

        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x1)

    return SecantResult(root=x1, iterations=max_iter, converged=False, history=history)