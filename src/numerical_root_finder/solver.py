from typing import Callable, Optional, Sequence

from .bisection import bisection_method
from .brent import brent_method
from .newton import newton_method
from .newton_system import newton_system
from .secant import secant_method
from .newton_system import newton_system 


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
    Solve a scalar nonlinear equation using the selected numerical method.

    Parameters
    ----------
    method : str
        Name of the numerical method to use. Supported values are
        ``"newton"``, ``"secant"``, ``"bisection"``, and ``"brent"``.
    f : Callable[[float], float]
        Scalar function.
    df : Callable[[float], float], optional
        Derivative of ``f`` when required.
    x0, x1 : float, optional
        Initial guesses.
    a, b : float, optional
        Initial bracketing interval.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    NewtonResult | SecantResult | BisectionResult | BrentResult
        Result object corresponding to the selected method.

    Raises
    ------
    ValueError
        If the supplied arguments are inconsistent with the selected
        method.
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
    Solve a system of nonlinear equations using the selected solver.

    Parameters
    ----------
    method : str
        Name of the solver. Currently only ``"newton"`` is supported.
    F : Callable[[np.ndarray], np.ndarray]
        Vector-valued nonlinear function.
    x0 : ArrayLike
        Initial approximation.
    jac : Callable, optional
        Analytic Jacobian.

    Returns
    -------
    NewtonSystemResult
        Structured result returned by the multidimensional Newton solver.

    Raises
    ------
    ValueError
        If an unsupported method is requested.
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
