from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .newton_system import finite_difference_jacobian


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BroydenResult:
    """
    Result returned by Broyden's method.

    Attributes
    ----------
    root : numpy.ndarray
        Final approximation to the solution.
    converged : bool
        Whether the solver satisfied a convergence criterion.
    iterations : int
        Number of completed Broyden iterations.
    residual_norm : float
        Euclidean norm of the final residual.
    step_norm : float
        Euclidean norm of the final accepted step.
    residual_history : tuple[float, ...]
        Residual norm recorded at the initial point and after each step.
    message : str
        Human-readable explanation of why the solver stopped.
    """

    root: FloatArray
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    residual_history: tuple[float, ...]
    message: str


def broyden_system(
    F: Callable[[FloatArray], ArrayLike],
    x0: Sequence[float] | FloatArray,
    jac: Callable[[FloatArray], ArrayLike] | None = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 50,
    fd_method: str = "central",
    fd_eps: float = 1e-6,
    line_search: bool = True,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    ls_shrink: float = 0.5,
    ls_max_steps: int = 20,
    min_update_denom: float = 1e-14,
) -> BroydenResult:
    """
    Solve a square nonlinear system using Broyden's good method.

    Broyden's method avoids recomputing the full Jacobian at every
    iteration. It begins with an analytic or finite-difference Jacobian
    and updates that approximation using a rank-one secant correction.

    Parameters
    ----------
    F : Callable[[numpy.ndarray], ArrayLike]
        Vector-valued function defining the nonlinear system.
    x0 : Sequence[float] or numpy.ndarray
        Initial approximation to the solution.
    jac : Callable[[numpy.ndarray], ArrayLike], optional
        Initial analytic Jacobian. If omitted, a finite-difference
        approximation is computed at ``x0``.
    tol_f : float, optional
        Convergence tolerance applied to the residual norm.
    tol_x : float, optional
        Convergence tolerance applied to the accepted step norm.
    max_iter : int, optional
        Maximum number of Broyden iterations.
    fd_method : {"forward", "central"}, optional
        Finite-difference scheme used when ``jac`` is not supplied.
    fd_eps : float, optional
        Perturbation size used for finite differences.
    line_search : bool, optional
        Whether to use Armijo-style backtracking.
    alpha0 : float, optional
        Initial line-search step length.
    c1 : float, optional
        Armijo sufficient-decrease constant.
    ls_shrink : float, optional
        Multiplicative line-search reduction factor.
    ls_max_steps : int, optional
        Maximum number of backtracking reductions.
    min_update_denom : float, optional
        Minimum allowed value of ``s.T @ s`` for the rank-one update.

    Returns
    -------
    BroydenResult
        Structured result containing the solution estimate, convergence
        information, residual history, and stopping message.

    Notes
    -----
    The step ``p_k`` is obtained by solving

        B_k p_k = -F(x_k),

    where ``B_k`` approximates the Jacobian. After accepting the step

        s_k = x_{k+1} - x_k,

    the approximation is updated using Broyden's good update:

        B_{k+1}
        = B_k + ((y_k - B_k s_k) s_k^T) / (s_k^T s_k),

    where

        y_k = F(x_{k+1}) - F(x_k).
    """
    if tol_f <= 0:
        raise ValueError("tol_f must be positive.")
    if tol_x <= 0:
        raise ValueError("tol_x must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if fd_eps <= 0:
        raise ValueError("fd_eps must be positive.")
    if alpha0 <= 0:
        raise ValueError("alpha0 must be positive.")
    if not 0 < c1 < 1:
        raise ValueError("c1 must lie strictly between 0 and 1.")
    if not 0 < ls_shrink < 1:
        raise ValueError("ls_shrink must lie strictly between 0 and 1.")
    if ls_max_steps < 0:
        raise ValueError("ls_max_steps cannot be negative.")
    if min_update_denom <= 0:
        raise ValueError("min_update_denom must be positive.")

    x = np.asarray(x0, dtype=float).copy()

    if x.ndim != 1:
        raise ValueError("x0 must be a one-dimensional vector.")
    if x.size == 0:
        raise ValueError("x0 must contain at least one value.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain only finite values.")

    fx = np.asarray(F(x), dtype=float)

    if fx.ndim != 1:
        raise ValueError("F(x) must return a one-dimensional vector.")
    if fx.shape != x.shape:
        raise ValueError(
            "Broyden's method requires a square system: "
            "F(x) and x must have the same dimension."
        )
    if not np.all(np.isfinite(fx)):
        raise ValueError("F(x0) returned non-finite values.")

    residual_norm = float(np.linalg.norm(fx))
    residual_history = [residual_norm]

    if residual_norm <= tol_f:
        return BroydenResult(
            root=x,
            converged=True,
            iterations=0,
            residual_norm=residual_norm,
            step_norm=0.0,
            residual_history=tuple(residual_history),
            message="Converged: initial residual norm below tol_f.",
        )

    if jac is None:
        B = finite_difference_jacobian(
        F,
        x,
        fx=fx,
        method=fd_method,
        eps=fd_eps,
    )
    else:
        B = np.asarray(jac(x), dtype=float)
    if B.shape != (x.size, x.size):
        raise ValueError(
            f"Initial Jacobian must have shape {(x.size, x.size)}, "
            f"but received {B.shape}."
        )
    if not np.all(np.isfinite(B)):
        raise ValueError("Initial Jacobian contains non-finite values.")

    step_norm = 0.0

    for iteration in range(1, max_iter + 1):
        try:
            direction = np.linalg.solve(B, -fx)
        except np.linalg.LinAlgError:
            direction, *_ = np.linalg.lstsq(B, -fx, rcond=None)

        if not np.all(np.isfinite(direction)):
            return BroydenResult(
                root=x,
                converged=False,
                iterations=iteration - 1,
                residual_norm=residual_norm,
                step_norm=step_norm,
                residual_history=tuple(residual_history),
                message="Stopped: linear solve produced a non-finite step.",
            )

        alpha = alpha0
        trial_x = x + alpha * direction
        trial_fx = np.asarray(F(trial_x), dtype=float)
        trial_norm = float(np.linalg.norm(trial_fx))

        if line_search:
            target = (1.0 - c1 * alpha) * residual_norm

            for _ in range(ls_max_steps):
                if np.isfinite(trial_norm) and trial_norm <= target:
                    break

                alpha *= ls_shrink
                trial_x = x + alpha * direction
                trial_fx = np.asarray(F(trial_x), dtype=float)
                trial_norm = float(np.linalg.norm(trial_fx))
                target = (1.0 - c1 * alpha) * residual_norm

        if trial_fx.shape != fx.shape:
            raise ValueError(
                "F(x) changed output dimension during the iteration."
            )

        if not np.all(np.isfinite(trial_fx)):
            return BroydenResult(
                root=x,
                converged=False,
                iterations=iteration - 1,
                residual_norm=residual_norm,
                step_norm=step_norm,
                residual_history=tuple(residual_history),
                message="Stopped: function evaluation produced non-finite values.",
            )

        step = trial_x - x
        step_norm = float(np.linalg.norm(step))

        y = trial_fx - fx
        update_denom = float(step @ step)

        x = trial_x
        fx = trial_fx
        residual_norm = trial_norm
        residual_history.append(residual_norm)

        if residual_norm <= tol_f:
            return BroydenResult(
                root=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                step_norm=step_norm,
                residual_history=tuple(residual_history),
                message="Converged: residual norm below tol_f.",
            )

        if step_norm <= tol_x:
            return BroydenResult(
                root=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                step_norm=step_norm,
                residual_history=tuple(residual_history),
                message="Converged: step norm below tol_x.",
            )

        if update_denom <= min_update_denom:
            return BroydenResult(
                root=x,
                converged=False,
                iterations=iteration,
                residual_norm=residual_norm,
                step_norm=step_norm,
                residual_history=tuple(residual_history),
                message=(
                    "Stopped: step too small for a stable Broyden update."
                ),
            )

        correction = y - B @ step
        B = B + np.outer(correction, step) / update_denom

    return BroydenResult(
        root=x,
        converged=False,
        iterations=max_iter,
        residual_norm=residual_norm,
        step_norm=step_norm,
        residual_history=tuple(residual_history),
        message="Stopped: maximum number of iterations reached.",
    )