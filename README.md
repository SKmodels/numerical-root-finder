# Numerical Root Finder

![Tests](https://github.com/SKmodels/numerical-root-finder/actions/workflows/tests.yml/badge.svg)

A small Python library implementing classical numerical methods for solving nonlinear equations:  
```bash
f(x) = 0
```

Part of the **SKmodels** portfolio focused on scientific computing, numerical analysis and algorithm design.
---
## Implemented Methods
- **Newton-Raphson** - Quadratic convergence (order 2)
- **Bisection** - Guaranteed convergence (linear order 1)
- **Secant** - Superlinear convergence (~1.618)
- **Brent's Method** - Hybrid bracketing + Interpolation (robust and fast)

## Convergence Properties
- Newton - Derivative-based - Quadratics (p ~ 2)  
- Bisection - Bracketing (Robust) - Linear (p ~ 1)  
- Secant - Derivative-free - Superlinear (p ~ 1.618)  
- Brent's - Bracketing (robust) + inverse quadratic interpolation/secant (fast), with bisection fallback to guarantee convergence

### Definition of Convergence Order

The order of convergence \(p\) is defined by:
```bash
p = \lim_{n \to \infty}\frac{\log(e_{n+1}/e_n)}{\log(e_n/e_{n-1})}
```
where  
```bash
e_n = |x_n - r|
```

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

To run each method;

```bash
python -m [name]_usage
```
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
### Secant method
```python
from methods.secant import secant_method

def main():
    f = lambda x: x**2 - 2
    result = secant_method(f, x0=1.0, x1=2.0)

    print("Root:", result.root)
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
    print("Verification f(root):", f(result.root))
```
### Brent's method
```python
from methods.brent import brent_method

def main():
    f = lambda x: x**2 - 2
    result = brent_method(f, a=1.0, b=2.0)

    print("Root:", result.root)
    print("Iterations:", result.iterations)
    print("Converged:", result.converged)
```
### Convergence plot
```python
from methods.bisection import bisection_method
from methods.newton import newton_method
from methods.brent import brent_method
from methods.secant import secant_method
from pathlib import Path

def main() -> None:
  # Target: solve x^2 - 2 = 0 -> root = sqrt(2)
  true_root = math.sqrt(2)

  f= lambda x: x**2 - 2
  df = lambda x: 2 * x

  newton = newton_method(f, df, x0=1.5)
  bisect = bisection_method(f, a=1.0, b=2.0)
  brent = brent_method(f, a=1.0, b=2.0)
  secant = secant_method(f, x0=1.0, x1=2.0)
    
  # Your result objects have a `history` attribute that contains the sequence of approximations.
  newton_err = [abs(x - true_root) for x in newton.history]
  bisect_err = [abs(x - true_root) for x in bisect.history]
  brent_err = [abs(x - true_root) for x in brent.history]
  secant_err = [abs(x - true_root) for x in secant.history]

  plt.figure()
  plt.semilogy(range(len(newton_err)), newton_err, marker="o", label="Newton")
  plt.semilogy(range(len(bisect_err)), bisect_err, marker="o", label="Bisection")
  plt.semilogy(range(len(brent_err)), brent_err, marker="o", label="Brent")
  plt.semilogy(range(len(secant_err)), secant_err, marker="o", label="Secant")
  plt.xlabel("Iteration")
  plt.ylabel("Absolute Error |x - sqrt(2)|")
  plt.title("Convergence comparison")
  plt.legend()
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.tight_layout()
  project_root = Path(__file__).resolve().parents[1]
  docs_path = project_root / "docs"
  docs_path.mkdir(exist_ok=True)  # Ensure the docs directory exists
  out_file = docs_path / "convergence.png"
  print(f"Saving convergence plot to: {out_file}")
  plt.savefig(out_file, dpi=300, bbox_inches="tight")
  plt.show()
```
## Convergence Comparison

Run the comparison

```bash
python -m examples.plot_convergence
```

Example convergence behaviour for solving:

```bash
x^2 - 2 = 0
```
<img src="docs/convergence.png">

*Absolte error vs iteration (log scale).*

- Newton - faster near the root (quadratic) however, requires a derivative & a decent initial guess.
- Bisection - guaranteed convergence when the root is bracketed, but slower (linear).
- Secant - derivative-free & usually faster than bisection (superlinear), but not guaranteed convergence.
- Brent's method - achieves near-Newton speed while retaining bisection robustness.

## Performance Benchmark
 
Run the benchmark

```bash
python -m examples.benchmark_methods
``` 
Example solving \( x^2 - 2 = 0 \)

| Method     | Iterations | Final Error   | Time (s)  |
|------------|------------|---------------|-----------|
| Newton     | 4          | 0.00e+00      | 0.000014  |
| Secant     | 6          | 2.22e-16      | 0.000007  |
| Bisection  | 27         | 1.85e-09      | 0.000012  |
| Brent      | 5          | 4.17e-14      | 0.000028  |
  

### Unified Solver Interface

- The solve() function provides a unified interface:

```python
from methods.solver import solve

result = solve(
  method="secant",
  f=lambda x: x**2 - 2,
  x0=1.0
  x1=2.0
)
```
## Testing & CI

All methods are validated using:

- Analytical convergence order estimation
- Bracket shrink guarantees (bisection)
- Known closed-form roots (sqrt(2) example)
- Continuous integration via GitHub Actions

### Run test locally
```bash
pytest
```
## Design Philosophy 

- All solvers retur structures result objects containing: 
  - 'root'
  - 'converged'
  - 'iterations' - which is stored for convergence analysis
  - 'history' of approximations
- Convergence behaviour can be analysed programmatically 
- Unified solver interface ('solve()') supports multiple algorithms and explicitly implemented (with SciPy dependency)
- CI-tested using 'pytest' and GitHub Actions
- Emphasis on numerical stability and theoretical correctness

## Why This Project?

Root-finding is a foundational problem in scientific computing, numerical analysis, optimisation and machine learning.

This project was built as part of the SKmodels portfolio to develop a deeper understanding of numerical stability, convergence theory and algorithmic design. 