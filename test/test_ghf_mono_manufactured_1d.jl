using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "GlobalHFRep mono 1D manufactured convergence" begin
    T = Float64
    s0 = T(0.53)
    A = T(0.2)
    Tm = T(0.5)
    kappa = T(1.0)
    rhoL = T(2.0)
    V = kappa * A / rhoL
    dt = T(0.01)

    grid = CartesianGrid((0.0,), (1.0,), (129,))
    rep = GlobalHFRep(grid, x -> x - s0; axis=:x, interp=:linear)

    texact = (x, t) -> manufactured_planar_1d(
        x,
        t;
        s0=s0,
        V=V,
        Tm=Tm,
        A=A,
        kappa=kappa,
        rhoL=rhoL,
    ).T
    src = (x, t) -> A * V

    bc = BorderConditions(; left=Dirichlet((x, t) -> texact(x, t)), right=Dirichlet((x, t) -> texact(x, t)))
    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=src)
    opts = StefanOptions(
        ;
        scheme=:BE,
        reinit=false,
        coupling_max_iter=25,
        coupling_tol=1e-10,
        coupling_reltol=1e-10,
        coupling_damping=1.0,
    )
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        t0=0.0,
        uω0=(x, t) -> texact(x, t),
        uγ0=Tm,
        speed0=0.0,
    )

    step!(solver, dt)

    logs = solver.state.logs
    iters = logs[:coupling_iters_last]
    residuals = logs[:coupling_residuals_last]

    @test iters <= 25
    @test !isempty(residuals)
    @test residuals[end] < 1e-8
    @test residuals[min(end, 5)] <= 0.2 * residuals[1]

    s_exact = s0 + V * dt
    s_num = rep.xf[]
    @test isfinite(s_num)
    @test abs(s_num - s_exact) < 2e-2
end
