from __future__ import annotations

from dataclasses import dataclass 
from typing import Callable, List
from methods.newton import newton_method

Number = float
Func = Callable[[Number], Number]

@dataclass(frozen=True)
class NewtonResult:
    root: float
    iterations: int
    converged: bool
    history: List[float]

def newton_method(
    f: Func,
    df: Func,
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    min_derivative: float = 1e-12,
) -> NewtonResult:
    """ 
    Solve f(x) = 0 using the Newton-Raphson method. 
    
    Parameters
    ----------
    f: callable
        Function f(x).
    df: callable 
        Derivative f'(x).
    x0: float
        initial guess.
    tol: float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    min_derivative : float
        Threshold below which derivative is considered too small.

    Returns
    -------
    NewtonResult
        root, iterations, convergence flag, and iteration history.
    """

    x = float(x0)
    history: List[float] = [x]

    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < min_derivative: 
            return NewtonResult(
                root=x,
                iterations=k - 1,
                converged=False,
                history=history,
            )

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol: 
            return NewtonResult(
                root=x_new,
                iterations=k,
                converged=True,
                history=history,
            )

        x = x_new

    return NewtonResult(
        root=x,
        iterations=max_iter,
        converged=False,
        history=history,
    )

from methods.newton import newton_method
def main() -> None:
    # Example: solve x^2 - 2 = 0 (root = sqrt(2))
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
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
    print("Verification f(root):", f(result.root))

if __name__ == "__main__":
    main()

