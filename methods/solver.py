from typing import Callable, Optional, Sequence

from methods.bisection import bisection_method
from methods.newton import newton_method
from methods.secant import secant_method
from methods.brent import brent_method

from methods.newton_system import newton_system  # NEW


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
    """
    Solve a scalar root-finding problem f(x)=0 using the specified method.
    """
    method = method.lower()

    if method == "newton":
        if df is None or x0 is None:
            raise ValueError("Newton's method requires df and x0")
        return newton_method(f, df, x0, tol=tol, max_iter=max_iter)

    if method == "secant":
        if x0 is None or x1 is None:
            raise ValueError("Secant method requires x0 and x1")
        return secant_method(f, x0=x0, x1=x1, tol=tol, max_iter=max_iter)

    if method == "bisection":
        if a is None or b is None:
            raise ValueError("Bisection method requires a and b")
        return bisection_method(f, a=a, b=b, tol=tol, max_iter=max_iter)

    if method == "brent":
        if a is None or b is None:
            raise ValueError("Brent's method requires a and b")
        return brent_method(f, a=a, b=b, tol=tol, max_iter=max_iter)

    raise ValueError(
        f"Unknown method: {method}. Choose 'newton', 'secant', 'bisection', or 'brent'."
    )


def solve_system(
    method: str,
    F: Callable[["Sequence[float]"], "Sequence[float]"],
    x0: Sequence[float],
    jac: Optional[Callable[["Sequence[float]"], "Sequence[Sequence[float]]"]] = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 50,
    line_search: bool = True,
):
    """
    Solve a nonlinear system F(x)=0 for x in R^n using the specified method.

    Currently supported:
      - method="newton" (multidimensional Newton)

    Notes
    -----
    - If jac is None, a finite-difference Jacobian is used.
    - tol_f is the tolerance on ||F(x)||_2.
    - tol_x is the tolerance on ||dx||_2.
    """
    method = method.lower()

    if method == "newton":
        return newton_system(
            F,
            x0=x0,
            jac=jac,
            tol_f=tol_f,
            tol_x=tol_x,
            max_iter=max_iter,
            line_search=line_search,
        )

    raise ValueError(
        f"Unknown system method: {method}. Choose 'newton'."
    )
