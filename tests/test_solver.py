import math
from methods.solver import solve
import pytest

def test_solve_newton():
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x

    result = solve(method='newton', f=f, df=df, x0=1.5)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=1e-8)

def test_solve_bisection():
    f = lambda x: x**2 - 2

    result = solve(method='bisection', f=f, a=1.0, b=2.0)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=1e-8)

def test_solve_secant():
    f = lambda x: x**2 - 2

    result = solve(method='secant', f=f, x0=1.0, x1=2.0)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=1e-8)

def test_secant_missing_args():
    f = lambda x: x**2 - 2

    with pytest.raises(ValueError):
        solve(method='secant', f=f, x0=1.0)  # Missing x1

def test_solve_invalid_method():
    f = lambda x: x**2 - 2
    with pytest.raises(ValueError):
        solve(method='invalid_method', f=f)