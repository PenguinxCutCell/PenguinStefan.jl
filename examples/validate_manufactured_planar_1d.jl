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
levels = [129, 257, 513]

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
    opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=4, extend_iters=10)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    Texact(x, t) = manufactured_planar_1d(x, t; s0=s0, V=V, Tm=Tm, A=A, kappa=kappa, rhoL=rhoL).T
    solver = build_solver(prob; t0=t0, uω0=(x, t) -> Texact(x, t0), uγ0=Tm, speed0=V)

    h = 1 / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.25 * h, save_history=false)

    cap = out.solver.cache.model.cap_slab
    xs = [cap.C_ω[i][1] for i in 1:cap.ntotal]
    ws = copy(out.solver.cache.model.Vn1)
    us = copy(out.state.uω)
    s_num = interface_position_from_phi(rep)

    return (n=n, h=h, x=xs, w=ws, u=us, s=s_num)
end

results = map(run_case, levels)
ref = results[end]

function interp1(xnodes::AbstractVector{T}, vals::AbstractVector{T}, x::T) where {T}
    if x <= xnodes[1]
        return vals[1]
    elseif x >= xnodes[end]
        return vals[end]
    end
    i = clamp(searchsortedlast(xnodes, x), 1, length(xnodes) - 1)
    x0 = xnodes[i]
    x1 = xnodes[i + 1]
    θ = (x - x0) / (x1 - x0)
    return (1 - θ) * vals[i] + θ * vals[i + 1]
end

errs_T = Float64[]
errs_s = Float64[]
hs = Float64[]

for r in results[1:(end - 1)]
    e2 = 0.0
    for i in eachindex(r.x)
        wi = r.w[i]
        if isfinite(wi) && wi > 0
            d = r.u[i] - interp1(ref.x, ref.u, r.x[i])
            e2 += wi * d^2
        end
    end
    push!(errs_T, sqrt(e2))
    push!(errs_s, abs(r.s - ref.s))
    push!(hs, r.h)
end

ordT = fit_order(hs, errs_T)
ords = fit_order(hs, errs_s)

println("Manufactured planar 1D convergence (against finest-grid reference)")
println("n\th\tL2(T-Tref)\t|s-sref|")
for (k, r) in enumerate(results[1:(end - 1)])
    println("$(r.n)\t$(r.h)\t$(errs_T[k])\t$(errs_s[k])")
end
println("order L2(T): pairwise=$(ordT.pairwise), global=$(ordT.order_global)")
println("order interface: pairwise=$(ords.pairwise), global=$(ords.order_global)")
println("target: temperature > 1, interface ~ 1")
