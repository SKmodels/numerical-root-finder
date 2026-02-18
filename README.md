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

## Convergence Properties
- Bisection - Bracketing (Robust) - Linear (p ~ 1)  
- Secant - Derivative-free - Superlinear (p ~ 1.618)  
- Newton - Derivative-based - Quadratics (p ~ 2)  

Where the order of convergence \(p\) is defined by:  

\[
\lim_{n \ to \infty} \frac{\log(e_{n+1}/e_n)}{log(e_n/{n-1})}
\]

with \( e_n = |x_n - r| \)
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
### Convergence Comparison

Example convergence behaviour for solving x^2 - 2 = 0

![Convergence Plot](docs/convergence.png)

## Testing

All methods are validated using:

- Analytical convergence order estimation
- Bracket shrink guarantees (bisection)
- Known closed-form roots (sqrt(2) example)
- Continuous integration via GitHub Actions

### Running test locally
```bash
pytest
```
## Design Philosophy 

- All solvers retur structures result objects containing: 
  - 'root'
  - 'converged'
  - 'iterations'
  - 'history' of approximations
- Convergence behaviour can be analysed programmatically 
- Unified solver interface ('solve()') supports multiple algorithms
- CI-tested using 'pytest' and GitHub Actions.

## Why This Project?

Root-finding is a foundational problem in scientific computing, numerical analysis, optimisation and machine learning.

This project was built as part of the SKmodels portfolio to develop a deeper understanding of numerical stability, convergence theory and algorithmic design. 