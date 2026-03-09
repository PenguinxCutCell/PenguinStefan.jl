using DelimitedFiles

using CartesianGrids
using PenguinBCs
using PenguinStefan

T = Float64
s0 = T(0.50)
epsf = T(0.03)
A = T(0.25)
kappa = T(1.0)
rhoL = T(2.5)
V = kappa * A / rhoL
Tm = T(0.5)
Lx = T(1.0)
Ly = T(1.0)
tf = T(0.02)
dt = T(0.001)
ωy = 2T(pi) / Ly

xf_exact(y, t) = s0 + epsf * sin(ωy * y) + V * t
texact(x, y, t) = Tm + A * (xf_exact(y, t) - x)
src(x, y, t) = A * V + kappa * A * (ωy^2) * epsf * sin(ωy * y)

grid = CartesianGrid((0.0, 0.0), (Lx, Ly), (65, 65))
rep = GlobalHFRep(
    grid,
    (x, y) -> x - xf_exact(y, 0.0);
    axis=:x,
    interp=:linear,
    periodic_transverse=true,
)

bc = BorderConditions(
    ;
    left=Dirichlet((x, y, t) -> texact(0.0, y, t)),
    right=Dirichlet((x, y, t) -> texact(Lx, y, t)),
    bottom=Periodic(),
    top=Periodic(),
)
params = StefanParams(Tm, rhoL; kappa1=kappa, source1=src)
opts = StefanOptions(
    ;
    scheme=:BE,
    reinit=false,
    coupling_max_iter=20,
    coupling_tol=1e-7,
    coupling_reltol=1e-7,
    coupling_damping=1.0,
)
prob = StefanMonoProblem(grid, bc, params, rep, opts)

solver = build_solver(
    prob;
    t0=0.0,
    uω0=(x, y, t) -> texact(x, y, t),
    uγ0=Tm,
    speed0=0.0,
)

ynodes = collect(CartesianGrids.grid1d(grid, 2))
metrics = Vector{NTuple{5,Float64}}()
snapshots = Vector{Vector{Float64}}()
snapshot_times = Float64[]

push!(snapshots, copy(rep.xf))
push!(snapshot_times, 0.0)

nsteps = Int(round(tf / dt))
for n in 1:nsteps
    step!(solver, dt)
    t = solver.state.t
    logs = solver.state.logs
    iters = Float64(logs[:coupling_iters_last])
    rmax = Float64(logs[:coupling_residual_inf_last])
    vmax = maximum(abs.(solver.state.speed_full))
    xf_ref = [xf_exact(y, t) for y in ynodes]
    exf = maximum(abs.(rep.xf .- xf_ref))

    push!(metrics, (t, iters, rmax, vmax, exf))
    println("step=$(n) t=$(t) iters=$(Int(iters)) residual=$(rmax) max|V|=$(vmax) max|xf-xf_exact|=$(exf)")

    if n % 2 == 0 || n == nsteps
        push!(snapshots, copy(rep.xf))
        push!(snapshot_times, t)
    end
end

open("ghf_mono_2d_sinusoidal_front_metrics.csv", "w") do io
    println(io, "t,iters,residual_inf,max_abs_speed,xf_error_inf")
    for row in metrics
        println(io, join(row, ","))
    end
end

snapmat = hcat(snapshots...)
out = hcat(ynodes, snapmat)
open("ghf_mono_2d_sinusoidal_front_xf_snapshots.csv", "w") do io
    labels = ["y"; ["xf_t=$(t)" for t in snapshot_times]]
    println(io, join(labels, ","))
    writedlm(io, out, ',')
end

println("Wrote metrics to ghf_mono_2d_sinusoidal_front_metrics.csv")
println("Wrote xf snapshots to ghf_mono_2d_sinusoidal_front_xf_snapshots.csv")
