# TODO

## Backlog

- [x] Isotropic Gibbs-Thomson thermodynamics in Stefan (`GibbsThomson` with curvature + kinetic terms).
- [x] Explicit `C_γ` sampling path for Gibbs-Thomson (`phi_np1_pred` curvature + lagged `speed_full` kinetic term).
- [x] Mesh/time convergence smoke script for LevelSet and GlobalHF.
- [ ] Front Tracking interface representation.
- [ ] VOF interface representation.
- [ ] LevelSet nonlinear inner-iteration (Newton/fixed-point) coupling.
- [ ] Gibbs-Thomson anisotropy (surface tension / kinetic anisotropy).


## Status Matrix

| Area | Item | Status | Notes |
|---|---|---|---|
| Interface | LevelSet representation | Implemented | Explicit split stepping operational. |
| Interface | Global height function representation | Partial | Newton coupling implemented for graph-like interfaces; broader robustness still in progress. |
| Interface | Front Tracking representation | Missing | Planned roadmap item. |
| Interface | VOF representation | Missing | Planned roadmap item. |
| Coupling | GlobalHF Newton inner iteration | Implemented | Covered by existing tests and examples. |
| Coupling | LevelSet nonlinear inner iteration | Missing | Current LevelSet path is explicit only. |
| Thermodynamics | Isotropic Gibbs-Thomson | Implemented | Curvature + kinetic with `C_γ` sampling. |
| Thermodynamics | Anisotropic Gibbs-Thomson | Missing | Not in current implementation. |
