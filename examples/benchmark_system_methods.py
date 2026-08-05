import time
import numpy as np

from numerical_root_finder import (
    newton_system,
    broyden_system,
)


def F(v):
    x, y = v
    return np.array([
        x**2 + y**2 - 1,
        x - y,
    ])


def J(v):
    x, y = v
    return np.array([
        [2*x, 2*y],
        [1, -1],
    ])


def benchmark(name, func):
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    print(
        f"{name:<10}"
        f"{result.iterations:>6}"
        f"{result.residual_norm:>16.2e}"
        f"{elapsed:>14.6f}"
    )


print()

print("Method      Iterations    Residual Norm      Time (s)")
print("-" * 52)

benchmark(
    "Newton",
    lambda: newton_system(
        F,
        [0.8, 0.6],
        jac=J,
    ),
)

benchmark(
    "Broyden",
    lambda: broyden_system(
        F,
        [0.8, 0.6],
    ),
)