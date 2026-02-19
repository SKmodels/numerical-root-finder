from dataclasses import dataclass
from typing import Callable, List

@dataclass
class BrentResult:
    root: float
    iterations: int
    converged: bool
    history: List[float]
    a_final: float
    b_final: float

def brent_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> BrentResult:
    """ 
    Find a root of f(x) = 0 on [a,b] using Brent's method.  

    Requirements
    - f(a) and f(b) must have opposite signs (i.e. bracket a root).
    """
    fa = f(a)
    fb = f(b)

    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Brent's method requires f(a) and f(b) to have opposite signs (root must be bracketed)")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa

    history = []

    mflag = True
    s = b
    
    for iteration in range(1, max_iter + 1):
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + (b * fa * fc) / ((fb - fa) * (fb - fc)) + (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        condition1 = not ((3 * a + b) / 4 < s < b) if b > a else not (b < s < (3 * a + b) / 4)
        condition2 = mflag and abs(s - b) >= abs(b - c) / 2
        condition3 = not mflag and abs(s - b) >= abs(c - d) / 2 if 'd' in locals() else False
        condition4 = mflag and abs(b - c) < tol
        condition5 = not mflag and abs(c - d) < tol if 'd' in locals() else False

        if condition1 or condition2 or condition3 or condition4 or condition5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        history.append(s)

        d = c
        c = b
        fc = fb

        if (fa * fs < 0):
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(fb) < tol:
            return BrentResult(root=b, iterations=iteration, converged=True, history=history, a_final=a, b_final=b)
        
    return BrentResult(root=b, iterations=max_iter, converged=False, history=history, a_final=a, b_final=b)