import math 
import pytest

from methods.bisection import bisection_method

def test_bisection_sqrt2():
    f = lambda x: x**2 - 2

    result = bisection_method(f, a=1.0, b=2.0, tol=1e-10, max_iter=200)

    assert result.converged
    assert math.isclose(result.root, math.sqrt(2), rel_tol=0, abs_tol=1e-9)
    assert result.a_final <= result.root <= result.b_final

def test_bisection_requires_bracket():
    f = lambda x: x**2 + 1 # No real roots, so f(a) and f(b) will have the same sign (positive)

    with pytest.raises(ValueError):
        bisection_method(f, a=1.0, b=1.0)