using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "GlobalHFRep mono 2D graph residual convergence" begin
    T = Float64
    s0 = T(0.50)
    epsf = T(0.001)
    A = T(0.25)
    kappa = T(1.0)
    rhoL = T(2.5)
    V = kappa * A / rhoL
    Tm = T(0.5)
    dt = T(0.002)
    L = T(1.0)
    ω = 2T(pi) / L

    xf = (y, t) -> s0 + epsf * sin(ω * y) + V * t
    texact = (x, y, t) -> Tm + A * (xf(y, t) - x)
    src = (x, y, t) -> A * V + kappa * A * (ω^2) * epsf * sin(ω * y)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (49, 49))
    rep = GlobalHFRep(
        grid,
        (x, y) -> x - xf(y, 0.0);
        axis=:x,
        interp=:linear,
        periodic_transverse=true,
    )

    bc = BorderConditions(
        ;
        left=Dirichlet((x, y, t) -> texact(0.0, y, t)),
        right=Dirichlet((x, y, t) -> texact(1.0, y, t)),
        bottom=Periodic(),
        top=Periodic(),
    )
    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=src)
    opts = StefanOptions(
        ;
        scheme=:BE,
        reinit=false,
        coupling_max_iter=25,
        coupling_tol=1e-9,
        coupling_reltol=1e-9,
        coupling_damping=1.0,
    )
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        t0=0.0,
        uω0=(x, y, t) -> texact(x, y, t),
        uγ0=Tm,
        speed0=0.0,
    )

    step!(solver, dt)

    logs = solver.state.logs
    iters = logs[:coupling_iters_last]
    residuals = logs[:coupling_residuals_last]

    @test iters <= 25
    @test !isempty(residuals)
    @test all(isfinite, residuals)
    @test residuals[end] < 2e-6
    @test minimum(residuals) <= 0.5 * residuals[1]

    ynodes = collect(CartesianGrids.grid1d(grid, 2))
    xf_exact = [xf(y, dt) for y in ynodes]
    @test maximum(abs.(rep.xf .- xf_exact)) < 4e-2
end
