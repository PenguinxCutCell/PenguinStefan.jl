using Test
using StaticArrays
using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "thermo_bc=nothing regression (mono and diph)" begin
    # Mono
    gridm = CartesianGrid((0.0,), (1.0,), (97,))
    bcm = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0))
    optsm = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)

    repm_a = LevelSetRep(gridm, x -> x - 0.53)
    repm_b = LevelSetRep(gridm, x -> x - 0.53)
    p_default = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0)
    p_nothing = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0, thermo_bc=nothing)
    probm_a = StefanMonoProblem(gridm, bcm, p_default, repm_a, optsm)
    probm_b = StefanMonoProblem(gridm, bcm, p_nothing, repm_b, optsm)

    init_m = (x, t) -> 1.0 - ((1.0 - 0.5) / 0.53) * x
    solm_a = build_solver(probm_a; uω0=init_m, uγ0=0.5, speed0=0.0)
    solm_b = build_solver(probm_b; uω0=init_m, uγ0=0.5, speed0=0.0)
    step!(solm_a, 0.01)
    step!(solm_b, 0.01)

    @test maximum(abs.(solm_a.state.uω .- solm_b.state.uω)) < 1e-12
    @test maximum(abs.(solm_a.state.uγ .- solm_b.state.uγ)) < 1e-12
    @test maximum(abs.(vec(solm_a.state.speed_full .- solm_b.state.speed_full))) < 1e-12
    @test maximum(abs.(PenguinStefan.phi_values(repm_a) .- PenguinStefan.phi_values(repm_b))) < 1e-12

    # Diph
    gridd = CartesianGrid((0.0,), (1.0,), (97,))
    bcd = BorderConditions(; left=Dirichlet(0.5), right=Dirichlet(0.5))
    optsd = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)

    repd_a = LevelSetRep(gridd, x -> x - 0.53)
    repd_b = LevelSetRep(gridd, x -> x - 0.53)
    pd_default = StefanParams(0.5, 1.0; kappa1=1.0, kappa2=2.0, source1=0.0, source2=0.0)
    pd_nothing = StefanParams(0.5, 1.0; kappa1=1.0, kappa2=2.0, source1=0.0, source2=0.0, thermo_bc=nothing)
    probd_a = StefanDiphProblem(gridd, bcd, pd_default, repd_a, optsd)
    probd_b = StefanDiphProblem(gridd, bcd, pd_nothing, repd_b, optsd)

    sold_a = build_solver(probd_a; uω10=0.5, uγ10=0.5, uω20=0.5, uγ20=0.5, speed0=0.0)
    sold_b = build_solver(probd_b; uω10=0.5, uγ10=0.5, uω20=0.5, uγ20=0.5, speed0=0.0)
    step!(sold_a, 0.01)
    step!(sold_b, 0.01)

    @test maximum(abs.(sold_a.state.uω1 .- sold_b.state.uω1)) < 1e-12
    @test maximum(abs.(sold_a.state.uγ1 .- sold_b.state.uγ1)) < 1e-12
    @test maximum(abs.(sold_a.state.uω2 .- sold_b.state.uω2)) < 1e-12
    @test maximum(abs.(sold_a.state.uγ2 .- sold_b.state.uγ2)) < 1e-12
    @test maximum(abs.(vec(sold_a.state.speed_full .- sold_b.state.speed_full))) < 1e-12
    @test maximum(abs.(PenguinStefan.phi_values(repd_a) .- PenguinStefan.phi_values(repd_b))) < 1e-12
end

@testset "1D curvature-only Gibbs-Thomson matches constant-Tm path" begin
    grid = CartesianGrid((0.0,), (1.0,), (97,))
    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0))
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)

    rep_ref = LevelSetRep(grid, x -> x - 0.53)
    rep_gt = LevelSetRep(grid, x -> x - 0.53)

    p_ref = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0)
    p_gt = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(0.2; kinetic=0.0))

    prob_ref = StefanMonoProblem(grid, bc, p_ref, rep_ref, opts)
    prob_gt = StefanMonoProblem(grid, bc, p_gt, rep_gt, opts)
    init = (x, t) -> 1.0 - ((1.0 - 0.5) / 0.53) * x
    sol_ref = build_solver(prob_ref; uω0=init, uγ0=0.5, speed0=0.0)
    sol_gt = build_solver(prob_gt; uω0=init, uγ0=0.5, speed0=0.0)

    step!(sol_ref, 0.01)
    step!(sol_gt, 0.01)

    @test maximum(abs.(sol_ref.state.uω .- sol_gt.state.uω)) < 1e-10
    @test maximum(abs.(sol_ref.state.uγ .- sol_gt.state.uγ)) < 1e-10
    @test maximum(abs.(vec(sol_ref.state.speed_full .- sol_gt.state.speed_full))) < 1e-10
end

@testset "1D kinetic-only trace sign and Cγ sampling" begin
    grid = CartesianGrid((0.0,), (1.0,), (97,))
    rep = LevelSetRep(grid, x -> x - 0.53)
    bc = BorderConditions(; left=Dirichlet(0.5), right=Dirichlet(0.5))
    mu = 0.2
    params = StefanParams(0.5, 1.0; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(0.0; kinetic=mu))
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=2.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    solver = build_solver(prob; uω0=0.5, uγ0=0.5, speed0=0.15)
    t = solver.state.t
    dt = 0.01
    phi_n = PenguinStefan.phi_values(rep)
    phi_np1 = PenguinStefan.predict_phi(rep, solver.state.speed_full, t, dt)

    PenguinStefan.assemble_moving_diffusion_mono!(
        solver.cache,
        phi_n,
        phi_np1,
        solver.state.speed_full,
        solver.state.uω,
        solver.state.uγ,
        t,
        dt,
        prob,
    )

    cap = something(solver.cache.model.cap_slab)
    Cg = cap.C_γ
    Tg = PenguinStefan._gibbs_thomson_trace_values(prob, phi_np1, solver.state.speed_full, Cg, t + dt)
    mask = [all(isfinite, x) for x in Cg]
    Vg = PenguinStefan._sample_speed_at_interface(solver.state.speed_full, grid, Cg)
    Tmg = PenguinStefan._point_values(params.Tm, Cg, t + dt)
    T_expected = Tmg .- mu .* Vg
    @test maximum(abs.(Tg[mask] .- T_expected[mask])) < 1e-12

    trace_cb = solver.cache.model.bc_interface.value
    T_cb = [eval_bc(trace_cb, Cg[i], t + dt) for i in eachindex(Cg) if mask[i]]
    T_dir = [Tg[i] for i in eachindex(Cg) if mask[i]]
    @test maximum(abs.(T_cb .- T_dir)) < 1e-12
    @test all(T_dir .<= params.Tm .+ 1e-12) # minus sign: positive V lowers TΓ
end

@testset "2D mono stationary circle with Gibbs-Thomson" begin
    Tm = 0.5
    sigma = 0.08
    R = 0.35
    Teq = Tm - sigma / R

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (97, 97))
    rep = LevelSetRep(grid, (x, y) -> sqrt(x^2 + y^2) - R)
    bc = BorderConditions(; left=Dirichlet(Teq), right=Dirichlet(Teq), bottom=Dirichlet(Teq), top=Dirichlet(Teq))
    params = StefanParams(Tm, 1.0; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=0.0))
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=12, interface_band=3.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; uω0=Teq, uγ0=Teq, speed0=0.0)

    t = solver.state.t
    dt = 0.001
    phi_n = PenguinStefan.phi_values(rep)
    phi_np1 = PenguinStefan.predict_phi(rep, solver.state.speed_full, t, dt)
    PenguinStefan.assemble_moving_diffusion_mono!(
        solver.cache,
        phi_n,
        phi_np1,
        solver.state.speed_full,
        solver.state.uω,
        solver.state.uγ,
        t,
        dt,
        prob,
    )
    cap = something(solver.cache.model.cap_slab)
    Tg = PenguinStefan._gibbs_thomson_trace_values(prob, phi_np1, solver.state.speed_full, cap.C_γ, t + dt)
    gamma = cap.buf.Γ
    mask = [isfinite(gamma[i]) && gamma[i] > 1e-12 for i in eachindex(gamma)]
    Tgm = sum(Tg[mask]) / count(mask)
    @test isapprox(Tgm, Teq; atol=0.08, rtol=0.1)

    for _ in 1:4
        step!(solver, dt)
    end

    v_if = vec(solver.state.speed_full)[vec(solver.state.frozen_mask)]
    rnum = radius_from_volume(solver.cache.model.Vn1; dim=2)
    @test all(isfinite, v_if)
    @test abs(sum(v_if) / length(v_if)) < 0.1
    @test abs(rnum - R) < 0.08
end

@testset "2D diphasic stationary circle with Gibbs-Thomson" begin
    Tm = 0.5
    sigma = 0.06
    R = 0.35
    Teq = Tm - sigma / R

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (81, 81))
    rep = LevelSetRep(grid, (x, y) -> sqrt(x^2 + y^2) - R)
    bc = BorderConditions(; left=Dirichlet(Teq), right=Dirichlet(Teq), bottom=Dirichlet(Teq), top=Dirichlet(Teq))
    params = StefanParams(Tm, 1.0; kappa1=1.0, kappa2=1.7, source1=0.0, source2=0.0, thermo_bc=GibbsThomson(sigma; kinetic=0.0))
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=3.0)
    prob = StefanDiphProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; uω10=Teq, uγ10=Teq, uω20=Teq, uγ20=Teq, speed0=0.0)

    step!(solver, 0.001)
    cap1 = something(solver.cache.model.cap1_slab)
    gamma1 = cap1.buf.Γ
    mask = [isfinite(gamma1[i]) && gamma1[i] > 1e-12 for i in eachindex(gamma1)]
    @test isapprox(sum(solver.state.uγ1[mask]) / count(mask), Teq; atol=0.08, rtol=0.1)
    @test isapprox(sum(solver.state.uγ2[mask]) / count(mask), Teq; atol=0.08, rtol=0.1)
    @test maximum(abs.(solver.state.uγ1[mask] .- solver.state.uγ2[mask])) < 0.12

    v_if = vec(solver.state.speed_full)[vec(solver.state.frozen_mask)]
    @test all(isfinite, v_if)
    @test abs(sum(v_if) / length(v_if)) < 0.1
end

@testset "Gibbs-Thomson sign flip on phi -> -phi" begin
    Tm = 0.5
    sigma = 0.1
    R = 0.4
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (81, 81))
    rep = LevelSetRep(grid, (x, y) -> sqrt(x^2 + y^2) - R)
    bc = BorderConditions(; left=Dirichlet(Tm), right=Dirichlet(Tm), bottom=Dirichlet(Tm), top=Dirichlet(Tm))
    params = StefanParams(Tm, 1.0; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=0.0))
    prob = StefanMonoProblem(grid, bc, params, rep, StefanOptions(; reinit=false))

    xγ = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2pi; length=65)[1:end-1]]
    phi = PenguinStefan.phi_values(rep)
    speed = zeros(size(phi))
    Tplus = PenguinStefan._gibbs_thomson_trace_values(prob, phi, speed, xγ, 0.0)
    Tminus = PenguinStefan._gibbs_thomson_trace_values(prob, -phi, speed, xγ, 0.0)

    @test maximum(abs.(Tplus .+ Tminus .- 2Tm)) < 0.1
    @test sum(Tplus) / length(Tplus) < Tm
    @test sum(Tminus) / length(Tminus) > Tm
end

@testset "Gibbs-Thomson capillary+kinetic smoke test" begin
    Tm = 0.0
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (81, 81))
    rep = LevelSetRep(grid, (x, y) -> begin
        r = hypot(x, y)
        θ = atan(y, x)
        r0 = 0.32 * (1 + 0.1 * cos(4 * θ))
        r - r0
    end)

    bc = BorderConditions(; left=Dirichlet(0.12), right=Dirichlet(0.12), bottom=Dirichlet(0.12), top=Dirichlet(0.12))
    params = StefanParams(Tm, 1.0; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(0.03; kinetic=0.01))
    opts = StefanOptions(; scheme=:BE, reinit=true, reinit_every=2, extend_iters=12, interface_band=4.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; uω0=0.12, uγ0=Tm, speed0=0.0)

    phi0 = PenguinStefan.phi_values(rep)
    for _ in 1:6
        step!(solver, 8e-4)
    end
    phi1 = PenguinStefan.phi_values(rep)

    @test all(isfinite, solver.state.uω)
    @test all(isfinite, solver.state.uγ)
    @test all(isfinite, vec(solver.state.speed_full))
    @test all(isfinite, phi1)
    @test maximum(abs.(vec(solver.state.speed_full))) < 10.0
    @test maximum(abs.(phi1 .- phi0)) > 0.0
end
