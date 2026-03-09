using CartesianGrids
using PenguinBCs
using PenguinStefan

function interface_from_phi_1d(rep::LevelSetRep)
    ϕ = PenguinStefan.phi_values(rep)
    x = CartesianGrids.grid1d(rep.space_grid, 1)
    for i in 1:(length(x) - 1)
        a = ϕ[i]
        b = ϕ[i + 1]
        if a == 0
            return x[i]
        elseif a * b < 0
            θ = abs(a) / (abs(a) + abs(b))
            return (1 - θ) * x[i] + θ * x[i + 1]
        end
    end
    return NaN
end

function run_case(rep, grid, bc, params, opts, t0, tf, dt, texact, Tm)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; t0=t0, uω0=(x, t) -> texact(x, t), uγ0=Tm, speed0=0.0)

    wall = @elapsed solve!(solver, (t0, tf); dt=dt, save_history=false)
    cap = solver.cache.model.cap_slab
    terr = temperature_error_norms(solver.state.uω, (x, t) -> texact(x, t), cap, tf)
    s_vol = interface_position_from_volume(solver.cache.model.Vn1)
    return (solver=solver, wall=wall, terr=terr, s_vol=s_vol)
end

s0 = 0.53
V = 0.15
Tm = 0.5
A = 0.2
kappa = 1.0
rhoL = kappa * A / V
t0 = 0.0
tf = 0.03

grid = CartesianGrid((0.0,), (1.0,), (257,))
bc = BorderConditions(; left=Dirichlet((x, t) -> Tm + A * ((s0 + V * t) - x)), right=Dirichlet(Tm))
params = StefanParams(Tm, rhoL; kappa1=kappa, source1=(x, t) -> A * V)
texact = (x, t) -> manufactured_planar_1d(x, t; s0=s0, V=V, Tm=Tm, A=A, kappa=kappa, rhoL=rhoL).T

h = 1.0 / (grid.n[1] - 1)
dt = 0.2 * h

opts_ls = StefanOptions(; scheme=:CN, reinit=true, reinit_every=4, extend_iters=10)
opts_ghf = StefanOptions(
    ;
    scheme=:CN,
    reinit=false,
    coupling_max_iter=20,
    coupling_tol=1e-7,
    coupling_reltol=1e-7,
    coupling_damping=1.0,
)

rep_ls = LevelSetRep(grid, x -> x - s0)
rep_ghf = GlobalHFRep(grid, x -> x - s0; axis=:x, interp=:linear)

out_ls = run_case(rep_ls, grid, bc, params, opts_ls, t0, tf, dt, texact, Tm)
out_ghf = run_case(rep_ghf, grid, bc, params, opts_ghf, t0, tf, dt, texact, Tm)

s_ls_phi = interface_from_phi_1d(rep_ls)
s_ls_vol = out_ls.s_vol
s_ghf = rep_ghf.xf[]
s_ghf_vol = out_ghf.s_vol

println("Comparison: LevelSetRep vs GlobalHFRep (planar graph case)")
println("Final time tf=$(tf)")
println("")
println("LevelSetRep")
println("  runtime [s]            = $(out_ls.wall)")
println("  interface from phi      = $(s_ls_phi)")
println("  interface from volume   = $(s_ls_vol)")
println("  temperature L2 error    = $(out_ls.terr.L2)")
println("")
println("GlobalHFRep")
println("  runtime [s]             = $(out_ghf.wall)")
println("  interface from xf       = $(s_ghf)")
println("  interface from volume   = $(s_ghf_vol)")
println("  temperature L2 error    = $(out_ghf.terr.L2)")
println("  last coupling iters     = $(out_ghf.solver.state.logs[:coupling_iters_last])")
println("  last coupling residual  = $(out_ghf.solver.state.logs[:coupling_residual_inf_last])")
