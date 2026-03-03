using CartesianGrids
using PenguinBCs
using PenguinStefan
using SpecialFunctions

L = 2.0
s0 = 0.8
Tinf = 1.0
Tm = 0.0
kappa = 1.0

t0 = 1.0
tf = 1.2
levels = [33, 65, 129]

F0 = expint((s0^2) / 4)
rhoL = 4 * kappa * Tinf * exp(-(s0^2) / 4) / (s0^2 * F0)

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

    v0 = frank_disk_exact(R_exact(t0), t0; s0=s0, Tinf=Tinf, Tm=Tm).Vn
    solver = build_solver(prob; t0=t0, uω0=(x, y, t) -> Texact(x, y, t0), uγ0=Tm, speed0=v0)

    h = 2L / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.15 * h, save_history=false)

    cap = out.solver.cache.model.cap_slab
    x = CartesianGrids.grid1d(grid, 1)
    y = CartesianGrids.grid1d(grid, 2)
    u = reshape(out.state.uω, n, n)
    w = reshape(out.solver.cache.model.Vn1, n, n)
    R = radius_from_volume(out.solver.cache.model.Vn1; dim=2, complement=true, domain_measure=(2L)^2)
    return (n=n, h=h, x=x, y=y, u=u, w=w, R=R)
end

results = map(run_case, levels)
ref = results[end]

function interp2(xnodes::AbstractVector{T}, ynodes::AbstractVector{T}, vals::AbstractMatrix{T}, x::T, y::T) where {T}
    i = clamp(searchsortedlast(xnodes, x), 1, length(xnodes) - 1)
    j = clamp(searchsortedlast(ynodes, y), 1, length(ynodes) - 1)

    x0, x1 = xnodes[i], xnodes[i + 1]
    y0, y1 = ynodes[j], ynodes[j + 1]
    θx = (x - x0) / (x1 - x0)
    θy = (y - y0) / (y1 - y0)

    v00 = vals[i, j]
    v10 = vals[i + 1, j]
    v01 = vals[i, j + 1]
    v11 = vals[i + 1, j + 1]

    vx0 = (1 - θx) * v00 + θx * v10
    vx1 = (1 - θx) * v01 + θx * v11
    return (1 - θy) * vx0 + θy * vx1
end

errs_T = Float64[]
errs_R = Float64[]
hs = Float64[]

for r in results[1:(end - 1)]
    e2 = 0.0
    for j in eachindex(r.y), i in eachindex(r.x)
        wij = r.w[i, j]
        if isfinite(wij) && wij > 0
            d = r.u[i, j] - interp2(ref.x, ref.y, ref.u, r.x[i], r.y[j])
            e2 += wij * d^2
        end
    end
    push!(errs_T, sqrt(e2))
    push!(errs_R, abs(r.R - ref.R))
    push!(hs, r.h)
end

ordT = fit_order(hs, errs_T)
ordR = fit_order(hs, errs_R)

println("Frank disk convergence (against finest-grid reference)")
println("rhoL = $rhoL")
println("n\th\tL2(T-Tref)\t|R-Rref|")
for (k, r) in enumerate(results[1:(end - 1)])
    println("$(r.n)\t$(r.h)\t$(errs_T[k])\t$(errs_R[k])")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface radius: pairwise=$(ordR.pairwise), global=$(ordR.order_global)")
println("target: temperature > 1, interface ~ 1")
