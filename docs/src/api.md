**API & Types**

Primary exports

- Interface representation:
  - `AbstractInterfaceRep`, `LevelSetRep`
- Configuration:
  - `StefanParams`, `StefanOptions`
- Problem/state containers:
  - `StefanMonoProblem`, `StefanDiphProblem`
  - `StefanMonoState`, `StefanDiphState`
- Solver entry points:
  - `build_solver`, `step!`, `solve!`
- Exact/benchmark helpers:
  - `stefan_lambda_neumann`, `neumann_1d_similarity`, `manufactured_planar_1d`, `frank_disk_exact`
- Diagnostics and utilities:
  - `active_mask`, `phase_volume`, `interface_position_from_volume`, `radius_from_volume`
  - `temperature_error_norms`, `stefan_residual_metrics`, `fit_order`

For docstrings in the Julia REPL:

```julia
?StefanMonoProblem
?step!
?stefan_lambda_neumann
```
