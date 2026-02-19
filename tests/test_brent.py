import math
from methods.brent import brent_method

def test_brent_sqrt2():
    f = lambda x: x**2 - 2

    result = brent_method(f, a=1.0, b=2.0)

    assert result.converged
    assert abs(result.root - math.sqrt(2)) < 1e-8