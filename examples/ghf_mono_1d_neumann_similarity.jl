using CartesianGrids
using CartesianGeometry: geometric_moments, nan
using CartesianOperators: assembled_capacity
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
levels = [11, 21, 41, 81, 161, 321, 641]

s_exact(t) = neumann_1d_similarity(0.0, t; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).s
Texact(x, t) = neumann_1d_similarity(x, t; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).T

function run_case(n)
    grid = CartesianGrid((0.0,), (Lx,), (n,))
    rep = GlobalHFRep(grid, x -> x - s_exact(t0); axis=:x, interp=:linear)

    bc = BorderConditions(; left=Dirichlet(Ts), right=Dirichlet(Tm))
    params = StefanParams(Tm, rhoL; kappa1=α, source1=0.0)
    opts = StefanOptions(
        ;
        scheme=:CN,
        reinit=false,
        coupling_max_iter=20,
        coupling_tol=1e-7,
        coupling_reltol=1e-7,
        coupling_damping=1.0,
    )
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    v0 = neumann_1d_similarity(0.0, t0; Ts=Ts, Tm=Tm, α=α, Ste=Ste, λ=λ).Vn
    solver = build_solver(prob; t0=t0, uω0=(x, t) -> Texact(x, t0), uγ0=Tm, speed0=v0)

    h = Lx / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.1 * h, save_history=false)
    # Build final-time space moments/capacity (not cap_slab) for analytical sampling at C_ω.
    body_final = (x) -> x - rep.xf[]
    moms_final = geometric_moments(body_final, CartesianGrids.grid1d(grid), Float64, nan; method=:vofijul)
    cap_final = assembled_capacity(moms_final; bc=0.0)
    terr = temperature_error_norms(out.state.uω, (x, t) -> Texact(x, t), cap_final, tf)
    s_err = abs(rep.xf[] - s_exact(tf))

    it_hist = get(out.state.logs, :coupling_iters, Int[])
    it_avg = isempty(it_hist) ? 0.0 : sum(it_hist) / length(it_hist)
    return (
        n=n,
        h=h,
        errT=terr.L2,
        errs=s_err,
        it_avg=it_avg,
        it_last=get(out.state.logs, :coupling_iters_last, 0),
    )
end

results = map(run_case, levels)
hs = [r.h for r in results]
errs_T = [r.errT for r in results]
errs_s = [r.errs for r in results]

ordT = fit_order(hs, errs_T)
ords = fit_order(hs, errs_s)

println("GHF 1D Neumann similarity convergence (analytical at final-time C_ω)")
println("lambda = $λ")
println("n\th\tL2(T-Texact)\t|s-s_exact|\tavg iters\tlast iters")
for r in results
    println("$(r.n)\t$(r.h)\t$(r.errT)\t$(r.errs)\t$(r.it_avg)\t$(r.it_last)")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface: pairwise=$(ords.pairwise), global=$(ords.order_global)")
