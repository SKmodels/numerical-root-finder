from __future__ import annotations

from dataclasses import dataclass
from logging import root
from typing import Callable, List

Number = float
Func = Callable[[Number], Number]

@dataclass(frozen=True)
class BisectionResult:
    root: float
    iterations: int
    converged: bool
    history: List[float]
    a_final: float
    b_final: float


def bisection_method(
    f: Func,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> BisectionResult:
    """
    Solve a scalar nonlinear equation using the bisection method.

    Parameters
    ----------
    f : Callable[[float], float]
        Scalar function for which a root is sought.
    a : float
        Left endpoint of the initial bracketing interval.
    b : float
        Right endpoint of the initial bracketing interval.
    tol : float, optional
        Desired interval width for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    BisectionResult
        Structured result containing the computed root, convergence
        status, iteration count, approximation history, and final
        bracketing interval.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` do not have opposite signs.

    Notes   
    -----
    The bisection method is guaranteed to converge provided the initial
    interval brackets a root and the function is continuous.
    """
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return BisectionResult(root=a, iterations=0, converged=True, history=[a], a_final=a, b_final=a)
    if fb == 0.0:
        return BisectionResult(root=b, iterations=0, converged=True, history=[b], a_final=b, b_final=b)
    if fa * fb > 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs (root must be bracketed)")

    history: List[float] = []
    left, right = a, b
    fleft = fa
    root = (left + right) / 2.0
    
    for k in range(1, max_iter + 1):
        root = (left + right) / 2.0
        history.append(root)

        fm = f(root)

        # Stop Conditions
        if abs(fm) <= tol or (right - left) / 2.0 <= tol:
            return BisectionResult(root=root, iterations=k, converged=True, history=history, a_final=left, b_final=right)
        
        # Keep the subinterval that contains the root
        if fm * fleft < 0:
            right = root
        else:
            left = root
            fleft = fm

    return BisectionResult(root=root, iterations=max_iter, converged=False, history=history, a_final=left, b_final=right)