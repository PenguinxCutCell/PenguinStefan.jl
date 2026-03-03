using Test
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "Mono Stefan 1D speed sign and Γ-normalization" begin
    grid = CartesianGrid((0.0,), (1.0,), (129,))
    s = 0.53
    rep = LevelSetRep(grid, x -> x - s)

    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0))
    params = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0)
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(
        prob;
        t0=0.0,
        uω0=(x, t) -> 1.0 - ((1.0 - params.Tm) / s) * x,
        uγ0=0.5,
        speed0=0.0,
    )

    ϕ0 = PenguinStefan.phi_values(rep)
    step!(solver, 0.01)

    state = solver.state
    idx = findall(state.frozen_mask)
    @test !isempty(idx)

    v_if = vec(state.speed_full)[idx]
    vmean = sum(v_if) / length(v_if)
    @test all(isfinite, v_if)
    @test vmean > 0.0
    v_expected = (1.0 - params.Tm) / (params.rhoL * s)
    @test isapprox(vmean, v_expected; atol=0.2)

    ϕ1 = PenguinStefan.phi_values(rep)
    @test sqrt(sum(abs2, ϕ1 .- ϕ0)) > 0.0
end

@testset "No-interface regression" begin
    grid = CartesianGrid((0.0,), (1.0,), (65,))
    rep = LevelSetRep(grid, x -> -1.0)

    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(1.0))
    params = StefanParams(1.0, 1.0; kappa1=1.0, source1=0.0)
    opts = StefanOptions(; scheme=:BE, reinit=false)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(prob; uω0=1.0, uγ0=1.0)
    ϕ0 = PenguinStefan.phi_values(rep)
    step!(solver, 0.02)

    @test maximum(abs.(vec(solver.state.speed_full))) < 1e-14
    @test count(solver.state.frozen_mask) == 0
    @test sqrt(sum(abs2, PenguinStefan.phi_values(rep) .- ϕ0)) < 1e-14
end
