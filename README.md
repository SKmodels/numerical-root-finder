# numerical-root-finder
Implementation of classical numerical methods for solving nonlinear equations \(f(x)=0\) in Python.
This project is part of SKmodels portfolio, focused on building strong foundations in scientific computing and numerical analysis. 

## Implemented Methods
- Newton-Raphson Method

## How it works
- Convergence criterion: |x_{n+1} - x-n| < tol
- Safety: derivative threshold to avoice unstable steps

## Roadmap
- Bisection method
- Secant method
- CLI (user chooses method / function)
- Unit tests with pytest
- Plot convergence (matplotlib)

## Example usage
```bash
python -m examples.example_usage


