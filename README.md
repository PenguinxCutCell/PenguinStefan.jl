# PenguinStefan.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinStefan.jl/dev)
![CI](https://github.com/PenguinxCutCell/PenguinStefan.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/PenguinStefan.jl/branch/main/graph/badge.svg)


`PenguinStefan.jl` is a cut-cell Stefan solver package built on `PenguinDiffusion.jl`, `CartesianGeometry.jl`, `CartesianOperators.jl`, and `PenguinSolverCore.jl`.

## Feature Status

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Monophasic Stefan | Implemented | `StefanMonoProblem`, `StefanMonoState`, `step!` |
| Models | Diphasic Stefan | Implemented | `StefanDiphProblem`, `StefanDiphState`, `step!` |
| Interface | Level-set representation | Implemented | `LevelSetRep` |
| Solver | End-to-end stepping | Implemented | `build_solver`, `step!`, `solve!` |
| Validation | 1D Neumann similarity | Implemented | `neumann_1d_similarity`, `stefan_lambda_neumann` |
| Validation | Manufactured planar 1D | Implemented | `manufactured_planar_1d` |
| Validation | 2D Frank disk benchmark | Implemented | `frank_disk_exact` |
| Diagnostics | Error and residual metrics | Implemented | `temperature_error_norms`, `stefan_residual_metrics`, `fit_order` |
| Thermodynamics | Gibbs-Thomson (`TΓ = Tm - σκκΓ - μΓVΓ`) | Implemented | `thermo_bc=GibbsThomson(...)` |
| Thermodynamics | Explicit level-set curvature + `C_γ` sampling | Implemented | `κΓ` from `phi_np1_pred`, `VΓ` from lagged `speed_full` |

## Examples

```bash
julia --project=. examples/mono_1d_similarity.jl
julia --project=. examples/mono_2d_circle_melt.jl
julia --project=. examples/diph_1d_stationary.jl
julia --project=. examples/mono_2d_circle_gibbs_thomson_equilibrium.jl
julia --project=. examples/mono_2d_perturbed_crystal_shrink_gibbs_thomson.jl
```

## Gibbs-Thomson Notes

`StefanParams` accepts an optional `thermo_bc` field. If set to `nothing`, the
existing constant-`Tm` behavior is unchanged.

For `thermo_bc = GibbsThomson(capillary; kinetic=...)`, the current explicit
time choice is:
- curvature term from predicted interface geometry `phi_np1_pred`,
- kinetic term from lagged signed speed field `state.speed_full`,
- all Gibbs-Thomson terms sampled at interface centroids `C_γ`.

## Documentation

- Dev docs: <https://PenguinxCutCell.github.io/PenguinStefan.jl/dev>
- Local build:

```bash
julia --project=docs docs/make.jl
```
