# Numerical Root Finder

![Tests](https://github.com/SKmodels/numerical-root-finder/actions/workflows/tests.yml/badge.svg)

A lightweight Python library implementing classical numerical methods for solving nonlinear equations:

<p align="center"><b>f</b>(<i>x</i>) = 0</p>

and nonlinear systems:

<p align="center"><b>F</b>(<i>x</i>) = 0, &nbsp; x ∈ ℝⁿ</p>

Part of the **SKmodels** portfolio focused on scientific computing, numerical analysis, and algorithm design.

---

## Features

- Scalar root-finding (bracketing and open methods)
- Multidimensional Newton solver for nonlinear systems
- Optional analytic or finite-difference Jacobians
- Armijo-style backtracking line search
- Convergence tracking and benchmarking
- Unified solver interface
- Fully tested with CI (pytest + GitHub Actions)

---

## Implemented Methods

- **Newton–Raphson (1D)** — quadratic convergence (order ≈ 2)
- **Bisection** — guaranteed linear convergence (order ≈ 1)
- **Secant** — superlinear convergence (order ≈ 1.618)
- **Brent’s Method** — robust hybrid bracketing/interpolation
- **Multidimensional Newton (Systems)**

---

## Convergence Theory

### Order of Convergence

The order of convergence \( p \) is defined as:

<p align="center">
<b>p</b> =
lim<sub>n → ∞</sub>
log(e<sub>n+1</sub> / e<sub>n</sub>)
/
log(e<sub>n</sub> / e<sub>n−1</sub>)
</p>

where

<p align="center">
e<sub>n</sub> = |x<sub>n</sub> − r|
</p>

Different methods exhibit different convergence behaviour depending on smoothness assumptions and derivative availability.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/SKmodels/numerical-root-finder.git
cd numerical-root-finder
```

### Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Example Usage

Run example scripts from the project root:

```bash
python -m examples.[name]_usage
```

---

### Newton–Raphson (1D)

```python
from methods.newton import newton_method

f = lambda x: x**2 - 2
df = lambda x: 2*x

result = newton_method(f, df, x0=1.5)

print("Root:", result.root)
print("Converged:", result.converged)
```

---

### Multidimensional Newton (Systems)

```python
import numpy as np
from methods.solver import solve_system

def F(v):
    x, y = v
    return np.array([
        x**2 + y**2 - 1.0,
        x - y
    ])

def J(v):
    x, y = v
    return np.array([
        [2*x, 2*y],
        [1.0, -1.0]
    ])

res = solve_system(
    method="newton",
    F=F,
    x0=[0.8, 0.6],
    jac=J
)

print("Root:", res.root)
print("Converged:", res.converged)
```

This example computes the intersection of the unit circle and the line \( x = y \).

The solver computes updates by solving:

```
J(x_k) Δx_k = -F(x_k)
```

and updates:

```
x_{k+1} = x_k + α Δx_k
```

where \( α \) is determined via Armijo-style backtracking line search.

---

### Bisection Method

```python
from methods.bisection import bisection_method

f = lambda x: x**2 - 2

result = bisection_method(f, a=1.0, b=2.0)

print("Root:", result.root)
print("Converged:", result.converged)
```

---

### Secant Method

```python
from methods.secant import secant_method

f = lambda x: x**2 - 2

result = secant_method(f, x0=1.0, x1=2.0)

print("Root:", result.root)
print("Iterations:", result.iterations)
print("Converged:", result.converged)
```

---

### Brent’s Method

```python
from methods.brent import brent_method

f = lambda x: x**2 - 2

result = brent_method(f, a=1.0, b=2.0)

print("Root:", result.root)
print("Iterations:", result.iterations)
print("Converged:", result.converged)
```

---

## Convergence Comparison

Run:

```bash
python -m examples.plot_convergence
```

Example solving:

<p align="center"><i>x</i><sup>2</sup> − 2 = 0</p>

<img src="docs/convergence.png">

*Absolute error vs iteration (log scale).*

- **Newton** — quadratic convergence near the root (requires derivative & good initial guess).
- **Bisection** — guaranteed convergence when bracketing a root.
- **Secant** — derivative-free and typically faster than bisection.
- **Brent’s method** — near-Newton speed with bracketing robustness.

---

## Performance Benchmark

Run:

```bash
python -m examples.benchmark_methods
```

Example for solving \( x^2 - 2 = 0 \):

| Method     | Iterations | Final Error | Time (s)  |
|------------|------------|-------------|-----------|
| Newton     | 4          | 0.00e+00    | 0.000014  |
| Secant     | 6          | 2.22e-16    | 0.000007  |
| Bisection  | 27         | 1.85e-09    | 0.000012  |
| Brent      | 5          | 4.17e-14    | 0.000028  |

---

## Unified Solver Interface

Scalar problems:

```python
from methods.solver import solve

result = solve(
    method="secant",
    f=lambda x: x**2 - 2,
    x0=1.0,
    x1=2.0
)

print(result.root)
```

Multidimensional systems:

```python
from methods.solver import solve_system

result = solve_system(
    method="newton",
    F=F,
    x0=[0.8, 0.6],
    jac=J
)
```

---

## Testing & Continuous Integration

All methods are validated using:

- Known closed-form roots (e.g., √2 benchmark)
- Convergence order estimation
- Bracket shrink guarantees (bisection)
- Multidimensional system verification
- Continuous integration via GitHub Actions

Run tests locally:

```bash
pytest
```

---

## Design Philosophy

- All solvers return structured result objects containing:
  - `root`
  - `converged`
  - `iterations`
  - `history` of approximations
- Convergence behaviour can be analysed programmatically
- Unified solver interface (`solve()` / `solve_system()`)
- Emphasis on numerical stability and theoretical correctness
- Minimal dependencies

---

## Why This Project?

Root-finding is foundational to:

- Scientific computing
- Numerical analysis
- Optimisation
- Computational physics
- Machine learning

This project was built as part of the **SKmodels** portfolio to develop deeper intuition for convergence theory, numerical stability, and algorithmic design.

---

## Project Roadmap

This library is being developed as a progressively more advanced nonlinear solver toolkit, with emphasis on numerical stability, convergence theory, and scientific computing applications.

### Phase 1 — Strengthen Nonlinear System Solvers

- [ ] Implement **Broyden’s Method** (quasi-Newton for systems)
  - Jacobian-free update
  - Compare performance vs analytic Newton and finite-difference Newton
- [ ] Add Jacobian diagnostics
  - Condition number estimation
  - Singular/near-singular detection
  - Optional warnings for ill-conditioned systems
- [ ] Improve line search strategy
  - Parameter tuning exposure
  - Optional Wolfe conditions
- [ ] Expand system-level test coverage

---

### Phase 2 — Applied Nonlinear Modelling Examples

Add real-world nonlinear systems to demonstrate solver robustness:

- [ ] Curve/surface intersection problems
- [ ] Chemical equilibrium equations
- [ ] Nonlinear circuit equations
- [ ] Mechanical equilibrium systems
- [ ] Logistic map fixed points
- [ ] Small nonlinear ODE shooting problem

Goal: transition from abstract mathematical examples to applied computational modelling.

---

### Phase 3 — Higher-Dimensional & Performance Scaling

- [ ] Benchmark scaling with dimension (n = 2 → 10+)
- [ ] Compare analytic vs finite-difference Jacobian cost
- [ ] Profile line search impact
- [ ] Add structured performance reporting

---

### Phase 4 — Advanced Solver Techniques

- [ ] Trust-region Newton methods
- [ ] Sparse Jacobian support
- [ ] Inexact Newton methods
- [ ] Preconditioning strategies
- [ ] Hybrid globalisation techniques

---

### Phase 5 — Scientific Computing Extensions

- [ ] Link root-finding to optimisation frameworks
- [ ] Extend toward nonlinear least squares
- [ ] Add automatic differentiation support
- [ ] Explore PDE-inspired nonlinear solves

---

## Long-Term Vision

Develop this project into a compact but rigorous nonlinear solver framework suitable for:

- Scientific computing
- Computational physics
- Numerical optimisation
- Applied mathematics research
- Scientific machine learning

The goal is not just implementation, but demonstrable understanding of convergence theory, numerical stability, and algorithmic design.
