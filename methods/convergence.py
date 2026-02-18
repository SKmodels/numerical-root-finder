import math 
from typing import List

def estimate_order(errors: List[float], eps: float = 1e-14) -> float:
    """ Estimate the order of convergence using classical 3-point formula. p = log(|e_{n+1}|/|e_n|) / log(|e_n|/|e_{n-1}|) """
    e = [x for x in errors if x > eps]

    if len(e) < 3:
        raise ValueError("Need at least 3 non-zero error values above the threshold to estimate order.")
    
    p_vals = []
    for i in range(2, len(e)):
        e_nm1, e_n, e_np1 = e[i-2], e[i-1], e[i]
        if e_n == 0 or e_nm1 == 0 or e_np1 == 0:
            continue
        denom = math.log(e_n / e_nm1)
        if denom == 0:
            continue
        p = math.log(e_np1 / e_n) / denom
        if math.isfinite(p):
            p_vals.append(p)

    if not p_vals:
        raise ValueError("Could not compute any valid order estimates from the error data.")
    
    p_vals.sort()
    return p_vals[len(p_vals) // 2]  # Return median to reduce sensitivity to outliers