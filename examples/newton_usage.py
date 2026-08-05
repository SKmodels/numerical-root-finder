import math

from numerical_root_finder import newton_method


def main() -> None:
    def f(x: float) -> float:
        return x**2 - 2

    def df(x: float) -> float:
        return 2 * x

    result = newton_method(
        f=f,
        df=df,
        x0=1.5,
        tol=1e-12,
        max_iter=50,
    )

    print("Finding root of f(x) = x^2 - 2 using Newton-Raphson\n")

    for i, x in enumerate(result.history):
        print(f"Iter {i:02d}: x = {x:.12f}")

    print("\nFinal Result")
    print("Root:", f"{result.root:.12f}")
    print("Expected:", f"{math.sqrt(2):.12f}")
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
    print("Verification f(root):", f(result.root))


if __name__ == "__main__":
    main()