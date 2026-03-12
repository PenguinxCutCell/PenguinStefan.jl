using CartesianGrids
using PenguinBCs
using PenguinStefan

s0 = 0.31
V = 0.2
Tm = 0.0
A = 0.2
kappa = 1.0
rhoL = kappa * A / V

t0 = 0.0
tf = 0.08
levels = [17, 33, 65, 129, 257, 513]

function interface_position_from_phi(rep::LevelSetRep)
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

function run_case(n)
    grid = CartesianGrid((0.0,), (1.0,), (n,))
    rep = LevelSetRep(grid, x -> x - s0)

    left_bc(x, t) = Tm + A * ((s0 + V * t) - x)
    bc = BorderConditions(; left=Dirichlet(left_bc), right=Dirichlet(Tm))

    source1(x, t) = A * V
    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=source1)
    opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=1, extend_iters=50)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    Texact(x, t) = manufactured_planar_1d(x, t; s0=s0, V=V, Tm=Tm, A=A, kappa=kappa, rhoL=rhoL).T
    solver = build_solver(prob; t0=t0, uω0=(x, t) -> Texact(x, t0), uγ0=Tm, speed0=V)

    h = 1 / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.25 * h, save_history=false)
    s_num = interface_position_from_phi(rep)

    cap = out.solver.cache.model.cap_slab
    terr = temperature_error_norms(out.state.uω, (x, t) -> Texact(x, t), cap, tf)
    s_err = abs(s_num - (s0 + V * tf))
    return (n=n, h=h, errT=terr.L2, errs=s_err)
end

results = map(run_case, levels)
hs = [r.h for r in results]
errs_T = [r.errT for r in results]
errs_s = [r.errs for r in results]

ordT = fit_order(hs, errs_T)
ords = fit_order(hs, errs_s)

println("Manufactured planar 1D convergence (analytical at final-time C_ω)")
println("n\th\tL2(T-Texact)\t|s-s_exact|")
for r in results
    println("$(r.n)\t$(r.h)\t$(r.errT)\t$(r.errs)")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface: pairwise=$(ords.pairwise), global=$(ords.order_global)")
println("note: this script now uses analytical-at-C_ω errors (stricter than finest-grid-reference comparison)")
