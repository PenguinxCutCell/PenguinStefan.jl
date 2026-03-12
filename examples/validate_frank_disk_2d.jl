using CartesianGrids
using PenguinBCs
using PenguinStefan
using SpecialFunctions

L = 2.0
s0 = 0.8
# For the exterior active phase (phi = R - r), use an undercooled far field
# so the signed front speed is consistent with the benchmark growth law.
Tinf = -1.0
Tm = 0.0
kappa = 1.0

t0 = 1.0
tf = 1.2
levels = [33, 65, 129]

F0 = expint((s0^2) / 4)
# Keep rhoL positive for Tinf < 0.
rhoL = -4 * kappa * Tinf * exp(-(s0^2) / 4) / (s0^2 * F0)

R_exact(t) = frank_disk_exact(0.0, t; s0=s0, Tinf=Tinf, Tm=Tm).R
Texact(x, y, t) = frank_disk_exact(hypot(x, y), t; s0=s0, Tinf=Tinf, Tm=Tm).T

function run_case(n)
    grid = CartesianGrid((-L, -L), (L, L), (n, n))
    rep = LevelSetRep(grid, (x, y) -> R_exact(t0) - hypot(x, y))

    bcfun(x, y, t) = Texact(x, y, t)
    bc = BorderConditions(
        ; left=Dirichlet(bcfun), right=Dirichlet(bcfun),
        bottom=Dirichlet(bcfun), top=Dirichlet(bcfun),
    )

    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=0.0)
    opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=4, extend_iters=12, interface_band=4.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    # frank_disk_exact returns radial outward speed; LevelSetRep uses signed
    # normal speed with n = grad(phi)/|grad(phi)|, so apply the sign here.
    v0 = -frank_disk_exact(R_exact(t0), t0; s0=s0, Tinf=Tinf, Tm=Tm).Vn
    solver = build_solver(prob; t0=t0, uω0=(x, y, t) -> Texact(x, y, t0), uγ0=Tm, speed0=v0)

    h = 2L / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.15 * h, save_history=false)

    cap = out.solver.cache.model.cap_slab
    terr = temperature_error_norms(out.state.uω, (x, y, t) -> Texact(x, y, t), cap, tf)
    R = radius_from_volume(out.solver.cache.model.Vn1; dim=2, complement=true, domain_measure=(2L)^2)
    R_err = abs(R - R_exact(tf))
    return (n=n, h=h, errT=terr.L2, errR=R_err)
end

results = map(run_case, levels)
hs = [r.h for r in results]
errs_T = [r.errT for r in results]
errs_R = [r.errR for r in results]

ordT = fit_order(hs, errs_T)
ordR = fit_order(hs, errs_R)

println("Frank disk convergence (analytical at final-time C_ω)")
println("rhoL = $rhoL")
println("n\th\tL2(T-Texact)\t|R-R_exact|")
for r in results
    println("$(r.n)\t$(r.h)\t$(r.errT)\t$(r.errR)")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface radius: pairwise=$(ordR.pairwise), global=$(ordR.order_global)")
println("note: this script now uses analytical-at-C_ω errors (stricter than finest-grid-reference comparison)")
