using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "GlobalHFRep diph coupling branch" begin
    grid = CartesianGrid((0.0,), (1.0,), (97,))
    rep = GlobalHFRep(grid, x -> x - 0.53; axis=:x, interp=:linear)

    bc = BorderConditions(; left=Dirichlet(0.5), right=Dirichlet(0.5))
    params = StefanParams(0.5, 1.0; kappa1=1.0, kappa2=2.0, source1=0.0, source2=0.0)
    opts = StefanOptions(
        ;
        scheme=:BE,
        reinit=false,
        coupling_max_iter=4,
        coupling_tol=1e-8,
        coupling_reltol=1e-8,
        coupling_damping=0.8,
    )
    prob = StefanDiphProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        uω10=0.5,
        uγ10=0.5,
        uω20=0.5,
        uγ20=0.5,
        speed0=0.0,
    )

    phi0 = PenguinStefan.phi_values(rep)
    xf0 = copy(rep.xf)

    step!(solver, 0.01)

    @test solver.state.t ≈ 0.01
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

    vmax = maximum(abs.(vec(solver.state.speed_full)))
    @test vmax < 1e-4
    @test maximum(abs.(rep.xf .- xf0)) < 1e-4
    @test maximum(abs.(PenguinStefan.phi_values(rep) .- phi0)) < 1e-4
end
