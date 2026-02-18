
from dataclasses import dataclass
from typing import Callable, List

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
    """
    x = float(x0)
    history: List[float] = [x]

    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < min_derivative: 
            return NewtonResult(x, k - 1, False, history)

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol:
            return NewtonResult(x_new, k, True, history)

        x = x_new

    return NewtonResult(x, max_iter, False, history)