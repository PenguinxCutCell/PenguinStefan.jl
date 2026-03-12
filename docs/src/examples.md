# Examples

Run from repository root:

```bash
julia --project=. examples/mono_1d_similarity.jl
```

Available scripts

- `examples/mono_1d_similarity.jl`
- `examples/mono_2d_circle_melt.jl`
- `examples/diph_1d_stationary.jl`
- `examples/validate_neumann_similarity_1d.jl`
- `examples/validate_manufactured_planar_1d.jl`
- `examples/validate_diph_balanced_flux_1d.jl`
- `examples/validate_frank_disk_2d.jl`

Test suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
