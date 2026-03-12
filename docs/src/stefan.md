# Theory and PDE

`PenguinStefan.jl` advances one-phase and two-phase Stefan problems on Cartesian cut-cell meshes. The solver couples heat diffusion with interface motion.

For each phase, temperature obeys a diffusion equation

```math
\partial_t T = \nabla \cdot (D \nabla T) + s,
```

with boundary/interface conditions provided through `PenguinBCs.jl` and the cut-cell operators from `CartesianGeometry.jl` and `CartesianOperators.jl`.

The interface speed is computed from Stefan balance (normal heat-flux jump) and used to update the geometric representation (`AbstractInterfaceRep`, e.g. `LevelSetRep`).

Implemented model families

- Monophasic Stefan: `StefanMonoProblem`, `StefanMonoState`, `step!`.
- Diphasic Stefan: `StefanDiphProblem`, `StefanDiphState`, `step!`.
- Similarity/manufactured references for validation in 1D and 2D.
