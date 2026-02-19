# tests/test_newton_system.py
import numpy as np

from methods.newton_system import newton_system
from methods.solver import solve_system

def test_newton_system_converges_with_analytic_jacobian():
    def F(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array([x*x + y*y - 1.0, x - y])

    def J(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array([[2.0*x, 2.0*y],
                         [1.0, -1.0]])

    res = newton_system(F, x0=[0.8, 0.6], jac=J, tol_f=1e-12, max_iter=50)
    assert res.converged
    assert np.linalg.norm(F(res.root)) < 1e-10
    assert np.allclose(res.root[0], res.root[1], atol=1e-9)


def test_newton_system_converges_with_fd_jacobian():
    def F(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array([x*x + y*y - 1.0, x - y])

    res = newton_system(F, x0=[0.8, 0.6], jac=None, tol_f=1e-10, max_iter=80)
    assert res.converged
    assert np.linalg.norm(F(res.root)) < 1e-8  # FD is a bit looser


def test_newton_system_raises_on_nonsquare_system():
    def F(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array([x + y])  # maps R^2 -> R^1

    try:
        newton_system(F, x0=[1.0, 2.0])
    except ValueError as e:
        assert "square" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError for non-square system.")

def test_solve_system_newton():
    def F(v):
        x, y = v
        return np.array([x*x + y*y - 1.0, x - y])

    def J(v):
        x, y = v
        return np.array([[2.0*x, 2.0*y],
                         [1.0, -1.0]])

    res = solve_system("newton", F, x0=[0.8, 0.6], jac=J)
    assert res.converged
    assert np.linalg.norm(F(res.root)) < 1e-10
