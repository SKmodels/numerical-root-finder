# Numerical Root Finder

![Tests](https://github.com/SKmodels/numerical-root-finder/actions/workflows/tests.yml/badge.svg)

A small Python library implementing classical numerical methods for solving nonlinear equations:  
**f(x) = 0**

Part of the SKmodels portfolio focused on scientific computing and numerical analysis.
---
## Implemented Methods
- Newton-Raphson (derivative-based, quadratic convergence)
- Bisection (bracketing method, guaranteed convergence)
- Secant (derative-free, superlinear convergence)
---
## Installation 
### Repository
```bash
git clone https://github.com/SKmodels/numerical-root-finder.git
cd numerical-root-finder
```
### Install runtime dependencies 
```bash
pip install -r requirements.txt
```
### Install development dependencies
```bash
pip install -r requirements-dev.txt 
```

## Example usage
### Newton-Raphson method
```python 
from methods.newton import newton_method
f = lambda x: x**2 - 2 
df = lambda x: 2*x

result = newton_method(f, df, x0=1.5)
print("Root:", result.root)
print("Converged:", result.converged)
```
### Bisection method
```python 
from methods.bisection import bisection_method
f = lambda x: x**2 - 2 

result = bisection_method(f, a=1.0, b=2.0)
print("Root:", result.root)
print("Converged:", result.converged)
```
### Convergence plot (Newton vs Bisection)
```bash
python -m examples.plot_convergence
```
### Secant method
```bash
python -m examples.secant_usage
```
### Running test
```bash
pytest
```
## Roadmap
- Secant method
- Convergence visualisation
- Unified solver interface
- Performance comparison benchmarks 