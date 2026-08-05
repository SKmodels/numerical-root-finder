import numpy as np

from numerical_root_finder import broyden_system


def main() -> None:
    def F(v):
        x, y = v
        return np.array([
            x**2 + y**2 - 1.0,
            x - y,
        ])

    result = broyden_system(
        F=F,
        x0=[0.8, 0.6],
        tol_f=1e-10,
        max_iter=50,
    )

    print("Solving:")
    print("x^2 + y^2 = 1")
    print("x = y")
    print()

    print("Root:", result.root)
    print("Converged:", result.converged)
    print("Iterations:", result.iterations)
    print("Residual norm:", result.residual_norm)
    print("Step norm:", result.step_norm)
    print("Message:", result.message)


if __name__ == "__main__":
    main()