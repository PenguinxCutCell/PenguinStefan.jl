using CartesianGrids
using CartesianGeometry: geometric_moments, nan
using CartesianOperators: assembled_capacity
using PenguinBCs
using PenguinStefan

# 1D planar interface translating at constant velocity.
# Reference profile in the active phase x < s(t) (shifted by s0):
#   T(x,t) = exp(V*(s(t)-x)) - 1 for x < s(t), 0 otherwise
#   s(t)   = s0 + V*t
#
# With phi = x - s (active phase x < s), the signed GHF speed is +V.

V = 1.0
s0 = 0.05
Tm = 0.0
rhoL = 1.0

dt_fixed = 1e-6
nsteps_fixed = 400
tf_fixed = nsteps_fixed * dt_fixed

levels_space = [65, 129, 257]

function s_exact(t)
    return s0 + V * t
end

function Texact(x, t)
    return x < s_exact(t) ? (exp(V * (s_exact(t) - x)) - 1.0) : 0.0
end

function run_case(n; dt=dt_fixed, nsteps=nsteps_fixed)
    Lx = 1.0
    grid = CartesianGrid((0.0,), (Lx,), (n,))
    rep = GlobalHFRep(grid, x -> x - s0; axis=:x, interp=:linear)

    bc = BorderConditions(; left=Dirichlet((x, t) -> Texact(x, t)), right=Dirichlet(0.0))
    params = StefanParams(Tm, rhoL; kappa1=1.0, source1=0.0)
    opts = StefanOptions(
        ;
        scheme=:CN,
        reinit=false,
        coupling_max_iter=20,
        coupling_tol=1e-8,
        coupling_reltol=1e-8,
        coupling_damping=1.0,
    )
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; t0=0.0, uω0=(x, t) -> Texact(x, 0.0), uγ0=Tm, speed0=V)

    xf0 = rep.xf[]
    step!(solver, dt)
    v0_num = (rep.xf[] - xf0) / dt
    v0_err = abs(v0_num - V)

    for _ in 2:nsteps
        step!(solver, dt)
    end

    tf = nsteps * dt
    body_final = (x) -> x - rep.xf[]
    moms_final = geometric_moments(body_final, CartesianGrids.grid1d(grid), Float64, nan; method=:vofijul)
    cap_final = assembled_capacity(moms_final; bc=0.0)
    terr = temperature_error_norms(solver.state.uω, (x, t) -> Texact(x, t), cap_final, tf)
    l1_mean = terr.weight > 0 ? terr.L1 / terr.weight : NaN

    h = Lx / (n - 1)
    return (n=n, h=h, tf=tf, v0_err=v0_err, L1=l1_mean, Linf=terr.Linf)
end

println("GHF 1D solidifying planar translation")
println("fixed dt = $dt_fixed, fixed steps = $nsteps_fixed")
println("")

rows_space = map(n -> run_case(n; dt=dt_fixed, nsteps=nsteps_fixed), levels_space)
ord_l1_space = fit_order([r.h for r in rows_space], [r.L1 for r in rows_space])
ord_li_space = fit_order([r.h for r in rows_space], [r.Linf for r in rows_space])
ord_v0_space = fit_order([r.h for r in rows_space], [r.v0_err for r in rows_space])

println("Grid refinement (fixed dt, fixed steps)")
println("n\th\tL1_mean(T)\tLinf(T)\t|V0_num - V|")
for r in rows_space
    println("$(r.n)\t$(r.h)\t$(r.L1)\t$(r.Linf)\t$(r.v0_err)")
end
println("order L1_mean(T): pairwise=$(ord_l1_space.pairwise), global=$(ord_l1_space.order_global)")
println("order Linf(T): pairwise=$(ord_li_space.pairwise), global=$(ord_li_space.order_global)")
println("order |V0_num-V|: pairwise=$(ord_v0_space.pairwise), global=$(ord_v0_space.order_global)")
println("")

dts = [4e-6, 2e-6, 1e-6]
n_time = 129
rows_time = map(dts) do dt
    nsteps = Int(round(tf_fixed / dt))
    run_case(n_time; dt=dt, nsteps=nsteps)
end
ord_l1_time = fit_order(dts, [r.L1 for r in rows_time])
ord_li_time = fit_order(dts, [r.Linf for r in rows_time])

println("Time refinement (fixed grid n=$n_time, fixed final time $tf_fixed)")
println("dt\tnsteps\tL1_mean(T)\tLinf(T)")
for (dt, r) in zip(dts, rows_time)
    nsteps = Int(round(tf_fixed / dt))
    println("$dt\t$nsteps\t$(r.L1)\t$(r.Linf)")
end
println("time order proxy L1_mean(T): pairwise=$(ord_l1_time.pairwise), global=$(ord_l1_time.order_global)")
println("time order proxy Linf(T): pairwise=$(ord_li_time.pairwise), global=$(ord_li_time.order_global)")
