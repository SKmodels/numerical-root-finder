import math

from methods.secant import secant_method

def test_secant_sqrt2():
    f = lambda x: x**2 - 2
    # 2 guesses near the root
    result = secant_method(f, x0=1.0, x1=2.0, tol=1e-10, max_iter=100)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=1e-10, abs_tol=0.0)
    assert result.iterations > 0