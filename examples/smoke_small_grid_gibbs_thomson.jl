using Statistics

using CartesianGrids
using PenguinBCs
using PenguinStefan

function run_circle_equilibrium_smoke()
    Tm = 0.5
    rhoL = 1.0
    sigma = 0.08
    R0 = 0.35
    Teq = Tm - sigma / R0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (49, 49))
    rep = LevelSetRep(grid, (x, y) -> hypot(x, y) - R0)
    bc = BorderConditions(; left=Dirichlet(Teq), right=Dirichlet(Teq), bottom=Dirichlet(Teq), top=Dirichlet(Teq))

    params = StefanParams(Tm, rhoL; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=0.0))
    opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8, interface_band=3.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; uω0=Teq, uγ0=Teq, speed0=0.0)

    dt = 1e-3
    nsteps = 4
    vmax = 0.0
    r0 = radius_from_volume(solver.cache.model.Vn1; dim=2)
    for _ in 1:nsteps
        step!(solver, dt)
        vmax = max(vmax, maximum(abs.(vec(solver.state.speed_full))))
    end
    r1 = radius_from_volume(solver.cache.model.Vn1; dim=2)

    println("[small circle equilibrium]")
    println("  t            = ", solver.state.t)
    println("  max |V|      = ", vmax)
    println("  radius drift = ", abs(r1 - r0))
end

function run_crystal_shrink_smoke()
    Tm = 0.0
    rhoL = 1.0
    kappa = 1.0
    sigma = 0.03
    mu = 0.01
    Tbox = 0.12

    R0 = 0.35
    eps_shape = 0.12
    m = 6
    phi_seed(x, y) = begin
        r = hypot(x, y)
        theta = atan(y, x)
        rseed = R0 * (1 + eps_shape * cos(m * theta))
        r - rseed
    end

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (65, 65))
    rep = LevelSetRep(grid, phi_seed)
    bc = BorderConditions(; left=Dirichlet(Tbox), right=Dirichlet(Tbox), bottom=Dirichlet(Tbox), top=Dirichlet(Tbox))

    params = StefanParams(Tm, rhoL; kappa1=kappa, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=mu))
    opts = StefanOptions(; scheme=:BE, reinit=true, reinit_every=2, extend_iters=10, interface_band=4.0)
    prob = StefanMonoProblem(grid, bc, params, rep, opts)
    solver = build_solver(prob; uω0=Tbox, uγ0=Tm, speed0=0.0)

    dt = 7.5e-4
    nsteps = 6
    req0 = radius_from_volume(solver.cache.model.Vn1; dim=2)
    rstd0 = begin
        cap = solver.cache.model.cap_slab
        rs = [hypot(x[1], x[2]) for x in cap.C_γ if all(isfinite, x)]
        std(rs)
    end

    vmax = 0.0
    for _ in 1:nsteps
        step!(solver, dt)
        vmax = max(vmax, maximum(abs.(vec(solver.state.speed_full))))
    end

    req1 = radius_from_volume(solver.cache.model.Vn1; dim=2)
    rstd1 = begin
        cap = solver.cache.model.cap_slab
        rs = [hypot(x[1], x[2]) for x in cap.C_γ if all(isfinite, x)]
        std(rs)
    end

    println("[small perturbed crystal]")
    println("  t            = ", solver.state.t)
    println("  req0, req1   = ", req0, ", ", req1)
    println("  rstd0, rstd1 = ", rstd0, ", ", rstd1)
    println("  max |V|      = ", vmax)
end

run_circle_equilibrium_smoke()
run_crystal_shrink_smoke()
