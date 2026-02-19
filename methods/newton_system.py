# methods/newton_system.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, List, Tuple

import numpy as np


Vector = np.ndarray
Matrix = np.ndarray


@dataclass(frozen=True)
class NewtonSystemResult:
    root: Vector
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    residual_history: Tuple[float, ...]
    message: str


def _as_float_vector(x0: Sequence[float] | np.ndarray) -> Vector:
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("x0 must be a non-empty vector.")
    return x


def finite_difference_jacobian(
    F: Callable[[Vector], Vector],
    x: Vector,
    fx: Optional[Vector] = None,
    *,
    method: str = "central",
    eps: float = 1e-6,
) -> Matrix:
    """
    Finite-difference approximation of Jacobian J(x) where J_ij = dF_i/dx_j.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    fx0 = np.asarray(F(x), dtype=float).reshape(-1) if fx is None else np.asarray(fx, dtype=float).reshape(-1)

    n = x.size
    m = fx0.size
    J = np.zeros((m, n), dtype=float)

    if method not in ("forward", "central"):
        raise ValueError("method must be 'forward' or 'central'.")

    # Scale step with magnitude of x to reduce cancellation issues
    h = eps * (1.0 + np.abs(x))

    for j in range(n):
        e = np.zeros_like(x)
        e[j] = 1.0

        if method == "forward":
            x1 = x + h[j] * e
            f1 = np.asarray(F(x1), dtype=float).reshape(-1)
            J[:, j] = (f1 - fx0) / h[j]
        else:
            x_plus = x + h[j] * e
            x_minus = x - h[j] * e
            f_plus = np.asarray(F(x_plus), dtype=float).reshape(-1)
            f_minus = np.asarray(F(x_minus), dtype=float).reshape(-1)
            J[:, j] = (f_plus - f_minus) / (2.0 * h[j])

    return J


def newton_system(
    F: Callable[[Vector], Vector],
    x0: Sequence[float] | np.ndarray,
    jac: Optional[Callable[[Vector], Matrix]] = None,
    *,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 50,
    line_search: bool = True,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    ls_shrink: float = 0.5,
    ls_max_steps: int = 20,
    fd_method: str = "central",
    fd_eps: float = 1e-6,
) -> NewtonSystemResult:
    """
    Solve a square nonlinear system F(x)=0 using Newton's method.

    Parameters
    ----------
    F:
        Function mapping R^n -> R^n.
    x0:
        Initial guess.
    jac:
        Optional Jacobian function J(x) with shape (n, n). If None, finite differences are used.
    tol_f:
        Convergence tolerance on ||F(x)||_2.
    tol_x:
        Convergence/stagnation tolerance on ||dx||_2.
    max_iter:
        Maximum Newton iterations.
    line_search:
        If True, uses backtracking line search (Armijo-style on ||F||^2).
    alpha0:
        Initial step scale for line search.
    c1:
        Armijo constant (small positive).
    ls_shrink:
        Backtracking shrink factor in (0,1).
    ls_max_steps:
        Max backtracking steps.
    fd_method, fd_eps:
        Finite-difference Jacobian settings (if jac is None).

    Returns
    -------
    NewtonSystemResult
    """
    x = _as_float_vector(x0)
    fx = np.asarray(F(x), dtype=float).reshape(-1)

    if fx.size != x.size:
        raise ValueError(f"System must be square: len(F(x))={fx.size} but len(x)={x.size}.")

    res_hist: List[float] = []
    fnorm = float(np.linalg.norm(fx, ord=2))
    res_hist.append(fnorm)

    if fnorm <= tol_f:
        return NewtonSystemResult(
            root=x,
            converged=True,
            iterations=0,
            residual_norm=fnorm,
            step_norm=0.0,
            residual_history=tuple(res_hist),
            message="Already converged at initial guess.",
        )

    last_step_norm = float("inf")

    for k in range(1, max_iter + 1):
        # Build Jacobian
        if jac is None:
            J = finite_difference_jacobian(F, x, fx=fx, method=fd_method, eps=fd_eps)
        else:
            J = np.asarray(jac(x), dtype=float)

        if J.shape != (x.size, x.size):
            raise ValueError(f"Jacobian must be shape {(x.size, x.size)}; got {J.shape}.")

        # Solve J dx = -F(x)
        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            # Fallback: least squares (handles near-singular Jacobian more gracefully)
            dx, *_ = np.linalg.lstsq(J, -fx, rcond=None)

        step_norm = float(np.linalg.norm(dx, ord=2))
        last_step_norm = step_norm

        # Stagnation / tiny step
        if step_norm <= tol_x:
            return NewtonSystemResult(
                root=x,
                converged=(fnorm <= tol_f),
                iterations=k - 1,
                residual_norm=fnorm,
                step_norm=step_norm,
                residual_history=tuple(res_hist),
                message="Step size below tol_x (stagnation or convergence).",
            )

        # Candidate update (with optional backtracking)
        alpha = alpha0
        x_new = x + alpha * dx
        fx_new = np.asarray(F(x_new), dtype=float).reshape(-1)
        fnorm_new = float(np.linalg.norm(fx_new, ord=2))

        if line_search:
            f2 = fnorm * fnorm
            target = (1.0 - c1 * alpha) * f2
            ls_steps = 0

            while (fnorm_new * fnorm_new > target) and (ls_steps < ls_max_steps):
                alpha *= ls_shrink
                x_new = x + alpha * dx
                fx_new = np.asarray(F(x_new), dtype=float).reshape(-1)
                fnorm_new = float(np.linalg.norm(fx_new, ord=2))
                target = (1.0 - c1 * alpha) * f2
                ls_steps += 1

        # Accept
        x, fx, fnorm = x_new, fx_new, fnorm_new
        res_hist.append(fnorm)

        if fnorm <= tol_f:
            return NewtonSystemResult(
                root=x,
                converged=True,
                iterations=k,
                residual_norm=fnorm,
                step_norm=last_step_norm,
                residual_history=tuple(res_hist),
                message="Converged: residual norm below tol_f.",
            )

    return NewtonSystemResult(
        root=x,
        converged=False,
        iterations=max_iter,
        residual_norm=fnorm,
        step_norm=last_step_norm if np.isfinite(last_step_norm) else float("nan"),
        residual_history=tuple(res_hist),
        message="Max iterations reached without convergence.",
    )
