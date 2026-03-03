using CartesianGrids
using PenguinBCs
using PenguinStefan

s0 = 0.53
Tm = 0.0
k1 = 1.0
k2 = 2.0
A1 = 0.2
A2 = k1 * A1 / k2

TL = Tm + A1 * s0
TR = Tm - A2 * (1 - s0)

t0 = 0.0
tf = 0.05

function run_case(n)
    grid = CartesianGrid((0.0,), (1.0,), (n,))
    rep = LevelSetRep(grid, x -> x - s0)

    bc = BorderConditions(; left=Dirichlet(TL), right=Dirichlet(TR))
    params = StefanParams(Tm, 1.0; kappa1=k1, kappa2=k2, source1=0.0, source2=0.0)
    opts = StefanOptions(; scheme=:BE, reinit=true, reinit_every=4, extend_iters=8)
    prob = StefanDiphProblem(grid, bc, params, rep, opts)

    u1(x, t) = x <= s0 ? (Tm + A1 * (s0 - x)) : Tm
    u2(x, t) = x >= s0 ? (Tm - A2 * (x - s0)) : Tm

    solver = build_solver(
        prob;
        t0=t0,
        uω10=(x, t) -> u1(x, t0),
        uγ10=Tm,
        uω20=(x, t) -> u2(x, t0),
        uγ20=Tm,
        speed0=0.0,
    )

    h = 1 / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.5 * h, save_history=false)

    Vfinal = out.solver.cache.model.V1n1
    s_num = interface_position_from_volume(Vfinal)
    vmax = maximum(abs.(vec(out.state.speed_full)))
    return (n=n, h=h, vmax=vmax, drift=abs(s_num - s0), s_num=s_num)
end

rows = map(run_case, [65, 129, 257])

println("Diph balanced-flux 1D validation")
println("n\th\tvmax\t|s-s0|")
for r in rows
    println("$(r.n)\t$(r.h)\t$(r.vmax)\t$(r.drift)")
end

ord_v = fit_order([r.h for r in rows], [r.vmax for r in rows])
ord_s = fit_order([r.h for r in rows], [r.drift for r in rows])
println("order vmax: pairwise=$(ord_v.pairwise), global=$(ord_v.order_global)")
println("order |s-s0|: pairwise=$(ord_s.pairwise), global=$(ord_s.order_global)")
