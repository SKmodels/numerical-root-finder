import numpy as np
import pytest

from numerical_root_finder import broyden_system, solve_system


def test_broyden_converges_with_analytic_jacobian():
    def F(v):
        x, y = v
        return np.array([
            x**2 + y**2 - 1.0,
            x - y,
        ])

    def J(v):
        x, y = v
        return np.array([
            [2.0 * x, 2.0 * y],
            [1.0, -1.0],
        ])

    result = broyden_system(
        F,
        x0=[0.8, 0.6],
        jac=J,
        tol_f=1e-10,
        max_iter=50,
    )

    expected = np.array([
        1.0 / np.sqrt(2.0),
        1.0 / np.sqrt(2.0),
    ])

    assert result.converged
    assert np.allclose(result.root, expected, atol=1e-8)
    assert result.residual_norm < 1e-10
    assert result.iterations > 0
    assert len(result.residual_history) == result.iterations + 1


def test_broyden_converges_with_finite_difference_jacobian():
    def F(v):
        x, y = v
        return np.array([
            x**2 + y**2 - 1.0,
            x - y,
        ])

    result = broyden_system(
        F,
        x0=[0.8, 0.6],
        jac=None,
        tol_f=1e-10,
        max_iter=50,
    )

    expected = np.array([
        1.0 / np.sqrt(2.0),
        1.0 / np.sqrt(2.0),
    ])

    assert result.converged
    assert np.allclose(result.root, expected, atol=1e-8)
    assert result.residual_norm < 1e-10


def test_broyden_detects_initial_solution():
    def F(v):
        return np.asarray(v, dtype=float)

    result = broyden_system(
        F,
        x0=[0.0, 0.0],
    )

    assert result.converged
    assert result.iterations == 0
    assert result.residual_norm == pytest.approx(0.0)
    assert result.step_norm == pytest.approx(0.0)
    assert result.message == "Converged: initial residual norm below tol_f."


def test_broyden_rejects_nonsquare_system():
    def F(v):
        x, y = v
        return np.array([
            x + y,
            x - y,
            x**2,
        ])

    with pytest.raises(ValueError, match="square system"):
        broyden_system(
            F,
            x0=[1.0, 1.0],
        )


def test_broyden_rejects_invalid_tolerances():
    def F(v):
        return np.asarray(v, dtype=float)

    with pytest.raises(ValueError, match="tol_f must be positive"):
        broyden_system(
            F,
            x0=[1.0],
            tol_f=0.0,
        )

    with pytest.raises(ValueError, match="tol_x must be positive"):
        broyden_system(
            F,
            x0=[1.0],
            tol_x=0.0,
        )

def test_solve_system_dispatches_to_broyden():
    def F(v):
        x, y = v
        return np.array([
            x**2 + y**2 - 1.0,
            x - y,
        ])

    result = solve_system(
        method="broyden",
        F=F,
        x0=[0.8, 0.6],
    )

    expected = np.array([
        1.0 / np.sqrt(2.0),
        1.0 / np.sqrt(2.0),
    ])

    assert result.converged
    assert np.allclose(result.root, expected, atol=1e-8)