# examples/newton_system_usage.py
import numpy as np
from methods.newton_system import newton_system


def F(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([
        x**2 + y**2 - 1.0,  # circle
        x - y,              # line
    ])


def J(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([
        [2.0*x, 2.0*y],
        [1.0,   -1.0],
    ])


if __name__ == "__main__":
    res = newton_system(F, x0=[0.8, 0.6], jac=J)
    print(res)
    # Expect root near [1/sqrt(2), 1/sqrt(2)] ~ [0.7071, 0.7071]
