using Test
using SpecialFunctions

using CartesianGrids
using PenguinBCs
using PenguinStefan

@testset "LevelSet ND planar manufactured short-run" begin
    s0 = 0.31
    V = 0.2
    Tm = 0.0
    A = 0.2
    kappa = 1.0
    rhoL = kappa * A / V

    t0 = 0.0
    tf = 0.02
    n = 65

    Texact(x, y, t) = manufactured_planar_1d(x, t; s0=s0, V=V, Tm=Tm, A=A, kappa=kappa, rhoL=rhoL).T
    s_exact(t) = s0 + V * t

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    rep = LevelSetRep(grid, (x, y) -> x - s0)

    left_bc(x, y, t) = Tm + A * (s_exact(t) - x)
    bc = BorderConditions(
        ; left=Dirichlet(left_bc),
        right=Dirichlet(Tm),
        bottom=Dirichlet((x, y, t) -> Texact(x, y, t)),
        top=Dirichlet((x, y, t) -> Texact(x, y, t)),
    )

    source1(x, y, t) = A * V
    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=source1)
    opts = StefanOptions(; scheme=:CN, reinit=false, extend_iters=12, interface_band=4.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; t0=t0, uω0=(x, y, t) -> Texact(x, y, t0), uγ0=Tm, speed0=V)

    h = 1 / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.2 * h, save_history=false)
    cap = out.solver.cache.model.cap_slab

    terr = temperature_error_norms(out.state.uω, (x, y, t) -> Texact(x, y, t), cap, tf)
    s_num = phase_volume(out.solver.cache.model.Vn1) # planar area = interface position * 1
    s_err = abs(s_num - s_exact(tf))

    @test isfinite(terr.L2)
    @test terr.L2 < 1e-4
    @test s_num > s0
    @test s_err < 2e-3
end

@testset "Frank disk sign-consistent growth sanity (LevelSet)" begin
    L = 2.0
    s0 = 0.8
    Tinf = -1.0
    Tm = 0.0
    kappa = 1.0

    t0 = 1.0
    tf = 1.04
    n = 49

    F0 = expint((s0^2) / 4)
    rhoL = -4 * kappa * Tinf * exp(-(s0^2) / 4) / (s0^2 * F0)

    R_exact(t) = frank_disk_exact(0.0, t; s0=s0, Tinf=Tinf, Tm=Tm).R
    Texact(x, y, t) = frank_disk_exact(hypot(x, y), t; s0=s0, Tinf=Tinf, Tm=Tm).T

    grid = CartesianGrid((-L, -L), (L, L), (n, n))
    rep = LevelSetRep(grid, (x, y) -> R_exact(t0) - hypot(x, y))

    bcfun(x, y, t) = Texact(x, y, t)
    bc = BorderConditions(
        ; left=Dirichlet(bcfun), right=Dirichlet(bcfun),
        bottom=Dirichlet(bcfun), top=Dirichlet(bcfun),
    )

    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=0.0)
    opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=4, extend_iters=10, interface_band=4.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)

    v0 = -frank_disk_exact(R_exact(t0), t0; s0=s0, Tinf=Tinf, Tm=Tm).Vn
    solver = build_solver(prob; t0=t0, uω0=(x, y, t) -> Texact(x, y, t0), uγ0=Tm, speed0=v0)

    R0 = radius_from_volume(solver.cache.model.Vn1; dim=2, complement=true, domain_measure=(2L)^2)
    h = 2L / (n - 1)
    out = solve!(solver, (t0, tf); dt=0.1 * h, save_history=false)
    R1 = radius_from_volume(out.solver.cache.model.Vn1; dim=2, complement=true, domain_measure=(2L)^2)

    @test R1 > R0
    @test abs(R1 - R_exact(tf)) < 5e-2
    @test all(isfinite, out.state.speed_full)
end
