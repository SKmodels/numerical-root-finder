import math
from methods.newton import newton_method

def test_newton_sqrt2():
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x

    result = newton_method(f, df, x0=1.5)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=1e-8)
    assert result.iterations > 0

def test_newton_nonconvergence():
    f = lambda x: x**3
    df = lambda x: 3 * x**2

    result = newton_method(f, df, x0=0.0, max_iter=5)

    assert not result.converged