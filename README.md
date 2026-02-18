# numerical-root-finder
Implementation of classical numerical methods for solving nonlinear equations in Python.
This project is part of SKmodels portfolio, focused on building strong foundations in scientific computing and numerical analysis. 

## Implemented Methods
Newton-Raphson Method (in progress)

## Example usage
```python
from methods.newton import newton_method

def f(x):
  return x**2 - 2

def df(x):
  return 2*x

root, iterations, converged = newton_method(f, df, x0=1.0)
print("Root:", root)

