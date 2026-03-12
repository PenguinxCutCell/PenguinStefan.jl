using CartesianGrids
using PenguinBCs
using PenguinStefan

Ts = 1.0
Tm = 0.0
α = 1.0
Ste = 1.0
λ = stefan_lambda_neumann(Ste)
rhoL = (Ts - Tm) / Ste

t0 = 5e-3
tf = 8e-2
Lx = 2.0
levels = [161, 321, 641]

s_exact(t) = neumann_1d_similarity(0.0, t; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).s
Texact(x, t) = neumann_1d_similarity(x, t; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).T

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
    grid = CartesianGrid((0.0,), (Lx,), (n,))
    rep = LevelSetRep(grid, x -> x - s_exact(t0))

    bc = BorderConditions(; left=Dirichlet(Ts), right=Dirichlet(Tm))
    params = StefanParams(Tm, rhoL; kappa1=α, source1=0.0)
    opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=1, extend_iters=10)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    v0 = neumann_1d_similarity(0.0, t0; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).Vn
    solver = build_solver(prob; t0=t0, uω0=(x, t) -> Texact(x, t0), uγ0=Tm, speed0=v0)

    h = Lx / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.1 * h, save_history=false)
    s_num = interface_position_from_phi(rep)

    cap = out.solver.cache.model.cap_slab
    terr = temperature_error_norms(out.state.uω, (x, t) -> Texact(x, t), cap, tf)
    s_err = abs(s_num - s_exact(tf))
    return (n=n, h=h, errT=terr.L2, errs=s_err)
end

results = map(run_case, levels)
hs = [r.h for r in results]
errs_T = [r.errT for r in results]
errs_s = [r.errs for r in results]

ordT = fit_order(hs, errs_T)
ords = fit_order(hs, errs_s)

println("1D Neumann similarity convergence (analytical at final-time C_ω)")
println("lambda = $λ")
println("n\th\tL2(T-Texact)\t|s-s_exact|")
for r in results
    println("$(r.n)\t$(r.h)\t$(r.errT)\t$(r.errs)")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface: pairwise=$(ords.pairwise), global=$(ords.order_global)")
println("note: this script now uses analytical-at-C_ω errors (stricter than finest-grid-reference comparison)")
