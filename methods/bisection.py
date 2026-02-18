from __future__ import annotations

from dataclasses import dataclass
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
    Find a root of f(x) = 0 on [a,b] using the bisection method.  

    Requirements
    - f(a) and f(b) must have opposite signs (i.e. bracket a root).
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
    fleft, fright = fa, fb
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
            fright = fm
        else:
            left = root
            fleft = fm

    return BisectionResult(root=root, iterations=max_iter, converged=False, history=history, a_final=left, b_final=right)