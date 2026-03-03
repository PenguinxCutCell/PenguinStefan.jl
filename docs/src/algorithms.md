**Numerical Algorithms & Routines**

This page summarizes the high-level routines used in a typical time step.

1) Build solver state

- Construct a problem (`StefanMonoProblem` or `StefanDiphProblem`) from grid, geometry, thermal parameters, and options.
- Create an initial state and linear solver cache using `build_solver`.

2) Advance one step (`step!`)

- Recompute cut-cell geometry and capacities from the current interface representation.
- Assemble diffusion operators for active cells/phases.
- Solve the diffusion subproblem(s) using `PenguinDiffusion.jl`/`PenguinSolverCore.jl`.
- Compute interface normal speed from Stefan flux balance.
- Advect/update interface representation and refresh diagnostics.

3) Time integration (`solve!`)

- Iterate `step!` over the requested `tspan`.
- Optionally collect history for post-processing and convergence studies.

Validation helpers

- `neumann_1d_similarity`, `manufactured_planar_1d`, and `frank_disk_exact` provide exact/benchmark fields.
- `temperature_error_norms`, `stefan_residual_metrics`, and `fit_order` quantify accuracy and order.
