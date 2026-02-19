import math
import time

from methods.bisection import bisection_method
from methods.brent import brent_method 
from methods.secant import secant_method
from methods.newton import newton_method

def benchmark():
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    true_root = math.sqrt(2)

    results= []

    # Newton
    start = time.perf_counter()
    newton = newton_method(f, df, x0=1.5)
    t = time.perf_counter() - start
    results.append(('Newton', newton.iterations, abs(newton.root - true_root), t))

    # Secant
    start = time.perf_counter()
    secant = secant_method(f, x0=1.0, x1=2.0)
    t = time.perf_counter() - start
    results.append(('Secant', secant.iterations, abs(secant.root - true_root), t))

    # Bisection
    start = time.perf_counter()
    bisection = bisection_method(f, a=1.0, b=2.0)
    t = time.perf_counter() - start
    results.append(('Bisection', bisection.iterations, abs(bisection.root - true_root), t))

    # Brent
    start = time.perf_counter()
    brent = brent_method(f, a=1.0, b=2.0)
    t = time.perf_counter() - start
    results.append(('Brent', brent.iterations, abs(brent.root - true_root), t))

    print("\nMethod      Iterations   Final Error      Time (s)")
    print("----------------------------------------------------")
    for name, iters, error, t in results:
        print(f"{name:<11} {iters:<12} {error:<16.2e} {t:.6f}")

if __name__ == "__main__":
    benchmark()