# Changelog

## Unreleased

### Added

- Broyden’s good method for square nonlinear systems.
- Analytic or finite-difference initial Jacobian support.
- Rank-one Jacobian approximation updates.
- Armijo-style backtracking line search.
- Public `broyden_system()` API.
- Broyden support through `solve_system(method="broyden", ...)`.
- Dedicated Broyden tests and usage example.

### Changed

- Adopted a `src` package layout.
- Replaced the generic `methods` namespace with `numerical_root_finder`.
- Updated tests, examples, and documentation to use the public package API. 

## v1.0.0

Initial public release.

### Features

- Newton-Raphson
- Secant
- Bisection
- Brent
- Multidimensional Newton
- Finite-difference Jacobians
- Analytic Jacobians
- Armijo backtracking
- Unified solver interface
- Convergence analysis
- Benchmarks
- GitHub Actions CI