# Numerical Root Finder
A small Python library implementing classical numerical methods for solving nonlinear equations: f(x) = 0
Part of the **SKmodels** portfolio focused on scientific computing and numerical analysis.
---
## Implemented Methods
- Newton-Raphson (derivative-based, quadratic convergence)
- Bisection (bracketing method, guaranteed convergence)
---
## Installation 
git clone https://github.com/SKmodels/numerical-root-finder.git
cd numerical-root-finder
pip install -r requirements.txt

## Example usage
### Newton method
python 
from methods.newton import newton_method
f = lambda x: x**2 - 2 
df = lambda x: 2*x

result = newton_method(f, df, x0=1.5)
print(result.root)

### Bisection method
python 
from methods.bisection import bisection_method
f = lambda x: x**2 - 2 

result = bisection_method(f, a=1.0, b=2.0)
print(result.root)

### Running test
pytest

## Roadmap
- Secant method
- Convergence visualisation
- Unified solver interface
- Performance comparison benchmarks 