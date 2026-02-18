import math 
from methods.bisection import bisection_method
from methods.newton import newton_method
from methods.secant import secant_method
from methods.convergence import estimate_order

def test_convergence_orders():
    true_root = math.sqrt(2)
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x

    # Newton's method
    newton = newton_method(f, df, x0=1.5)
    newton_err = [abs(x - true_root) for x in newton.history]
    p_newton = estimate_order(newton_err)
    assert 1.8 < p_newton < 2.2

    # Secant method
    secant = secant_method(f, x0=1.0, x1=2.0)
    secant_err = [abs(x - true_root) for x in secant.history]
    p_secant = estimate_order(secant_err)
    assert 1.5 < p_secant < 1.75

    # Bisection method
    a0, b0 = 1.0, 2.0
    bis = bisection_method(f, a=a0, b=b0)

    assert bis.converged
    assert bis.a_final <= true_root <= bis.b_final
    
    initial_width = b0 - a0
    final_width = bis.b_final - bis.a_final

    k = len(bis.history)
    steps = max(0, k - 1)
    assert final_width <= initial_width / (2 ** steps) + 1e-12
