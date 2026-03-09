using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "GlobalHFRep mono coupling branch" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 17))
    rep = GlobalHFRep(grid, (x, y) -> x - (0.53 + 0.02 * sin(2pi * y)); axis=:x, interp=:linear)

    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0), top=Neumann(0.0), bottom=Neumann(0.0))
    params = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0)
    opts = StefanOptions(
        ;
        scheme=:BE,
        reinit=false,
        extend_iters=8,
        interface_band=2.0,
        coupling_max_iter=4,
        coupling_tol=1e-8,
        coupling_reltol=1e-8,
        coupling_damping=0.8,
    )
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        t0=0.0,
        uω0=(x, y, t) -> 1.0 - ((1.0 - params.Tm) / 0.53) * x,
        uγ0=0.5,
        speed0=0.0,
    )

    phi0 = PenguinStefan.phi_values(rep)
    xf0 = copy(rep.xf)

    step!(solver, 0.002)

    @test solver.state.t ≈ 0.002
    @test coupling_mode(rep) == :ghf_newton
    @test haskey(solver.state.logs, :coupling_iters)
    @test haskey(solver.state.logs, :coupling_residual_inf)
    @test haskey(solver.state.logs, :coupling_residual_history)
    @test haskey(solver.state.logs, :coupling_residuals_last)
    @test !isempty(solver.state.logs[:coupling_iters])
    @test solver.state.logs[:coupling_iters_last] <= opts.coupling_max_iter
    rh = solver.state.logs[:coupling_residuals_last]
    @test !isempty(rh)
    @test all(isfinite, rh)
    @test maximum(abs.(rh)) < 1.0
    @test maximum(abs.(rep.xf .- xf0)) > 0.0
    @test maximum(abs.(PenguinStefan.phi_values(rep) .- phi0)) > 0.0
end
