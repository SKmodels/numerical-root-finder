from __future__ import annotations

import math

import matplotlib.pyplot as plt

from methods.bisection import bisection_method
from methods.newton import newton_method

def main() -> None:
    # Target: solve x^2 - 2 = 0 -> root = sqrt(2)
    true_root = math.sqrt(2)

    f= lambda x: x**2 - 2
    df = lambda x: 2 * x

    newton = newton_method(f, df, x0=1.5)
    bisect = bisection_method(f, a=1.0, b=2.0)
    
    # Your result objects have a `history` attribute that contains the sequence of approximations.
    newton_err = [abs(x - true_root) for x in newton.history]
    bisect_err = [abs(x - true_root) for x in bisect.history]

    plt.figure()
    plt.semilogy(range(len(newton_err)), newton_err, marker="o", label="Newton")
    plt.semilogy(range(len(bisect_err)), bisect_err, marker="o", label="Bisection")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error |x - sqrt(2)|")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()