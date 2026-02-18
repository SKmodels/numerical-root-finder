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
    Solve f(x) = 0 using the Secant method.

    Parameters
    ----------
    f: callable
        Function f(x).
    x0, x1: float
        Two initial guesses. (No need to bracket the root, but they should be close to it for good convergence.)
    tol: float
        Convergence tolerance on step size |x_{n+1} - x_n|.
    max_iter : int
        Maximum iterations.
    min_denom : float
        Minimum allowed |f(x1) - f(x0)| to avoid division by zero.

    Returns
    -------
    SecantResult
        root, iterations, converged and history.     
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