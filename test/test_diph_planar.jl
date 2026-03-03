using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "Diph Stefan planar stationary interface" begin
    grid = CartesianGrid((0.0,), (1.0,), (97,))
    rep = LevelSetRep(grid, x -> x - 0.53)

    bc = BorderConditions(; left=Dirichlet(0.5), right=Dirichlet(0.5))
    params = StefanParams(0.5, 1.0; kappa1=1.0, kappa2=2.0, source1=0.0, source2=0.0)
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)
    prob = StefanDiphProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        uω10=0.5,
        uγ10=0.5,
        uω20=0.5,
        uγ20=0.5,
        speed0=0.0,
    )

    ϕ0 = PenguinStefan.phi_values(rep)
    step!(solver, 0.01)

    vmax = maximum(abs.(vec(solver.state.speed_full)))
    @test vmax < 1e-5
    @test sqrt(sum(abs2, PenguinStefan.phi_values(rep) .- ϕ0)) < 1e-6
end
