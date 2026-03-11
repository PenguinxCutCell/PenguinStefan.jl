using DelimitedFiles
using Statistics

using CartesianGrids
using PenguinBCs
using PenguinStefan

Tm = 0.0
rhoL = 1.0
kappa = 1.0
sigma = 0.03
mu = 0.01
Tbox = 0.12

R0 = 0.35
eps_shape = 0.12
m = 6

function phi_seed(x, y)
    r = hypot(x, y)
    theta = atan(y, x)
    rseed = R0 * (1 + eps_shape * cos(m * theta))
    return r - rseed
end

function radius_std_from_cap(cap)
    rs = Float64[]
    @inbounds for x in cap.C_γ
        if all(isfinite, x)
            push!(rs, hypot(x[1], x[2]))
        end
    end
    return isempty(rs) ? NaN : std(rs)
end

grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (161, 161))
rep = LevelSetRep(grid, phi_seed)

bc = BorderConditions(; left=Dirichlet(Tbox), right=Dirichlet(Tbox), bottom=Dirichlet(Tbox), top=Dirichlet(Tbox))
params = StefanParams(Tm, rhoL; kappa1=kappa, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=mu))
opts = StefanOptions(; scheme=:BE, reinit=true, reinit_every=3, extend_iters=16, interface_band=4.0)
prob = StefanMonoProblem(grid, bc, params, rep, opts)

solver = build_solver(prob; uω0=Tbox, uγ0=Tm, speed0=0.0)

dt = 7.5e-4
nsteps = 80
save_every = 8

times = Float64[]
volumes = Float64[]
reqs = Float64[]
vmaxs = Float64[]
roundness = Float64[]

for step_id in 1:nsteps
    step!(solver, dt)
    cap = solver.cache.model.cap_slab
    vol = phase_volume(solver.cache.model.Vn1)
    req = radius_from_volume(solver.cache.model.Vn1; dim=2)
    vmax = maximum(abs.(vec(solver.state.speed_full)))
    rstd = radius_std_from_cap(cap)

    if step_id % save_every == 0 || step_id == nsteps
        push!(times, solver.state.t)
        push!(volumes, vol)
        push!(reqs, req)
        push!(vmaxs, vmax)
        push!(roundness, rstd)
        println("step=$step_id t=$(solver.state.t) volume=$vol req=$req max|V|=$vmax radius_std=$rstd")
    end
end

open("mono_2d_perturbed_crystal_shrink_gibbs_thomson_metrics.csv", "w") do io
    println(io, "t,phase_volume,equivalent_radius,max_abs_speed,radius_std")
    for i in eachindex(times)
        println(io, "$(times[i]),$(volumes[i]),$(reqs[i]),$(vmaxs[i]),$(roundness[i])")
    end
end

println("")
println("Mono 2D perturbed crystal shrink with Gibbs-Thomson + kinetic correction")
println("  initial equivalent radius ~ ", reqs[1])
println("  final equivalent radius   ~ ", reqs[end])
println("  initial radius std        = ", roundness[1])
println("  final radius std          = ", roundness[end])
println("  metrics file              = mono_2d_perturbed_crystal_shrink_gibbs_thomson_metrics.csv")
