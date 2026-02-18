from typing import Callable, Optional
from methods.bisection import bisection_method
from methods.newton import newton_method
from methods.secant import secant_method

def solve(
    method: str,
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]] = None,
    x0: Optional[float] = None,
    x1: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
):

    method = method.lower()
    
    if method == 'newton':
        if df is None or x0 is None:
            raise ValueError("Newton's method requires df and x0")
        return newton_method(f, df, x0, tol=tol, max_iter=max_iter)
    elif method == 'secant':
        if x0 is None or x1 is None:
            raise ValueError("Secant method requires x0 and x1")
        return secant_method(f, x0=x0, x1=x1, tol=tol, max_iter=max_iter) 
    elif method == 'bisection':
        if a is None or b is None:
            raise ValueError("Bisection method requires a and b")
        return bisection_method(f, a=a, b=b, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'newton', 'secant', or 'bisection'.")
    