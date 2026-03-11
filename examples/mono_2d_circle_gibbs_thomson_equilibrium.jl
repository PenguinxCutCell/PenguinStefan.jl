using CartesianGrids
using PenguinBCs
using PenguinStefan

Tm = 0.5
rhoL = 1.0
sigma = 0.08
R0 = 0.35
Teq = Tm - sigma / R0

grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (129, 129))
rep = LevelSetRep(grid, (x, y) -> sqrt(x^2 + y^2) - R0)

bc = BorderConditions(; left=Dirichlet(Teq), right=Dirichlet(Teq), bottom=Dirichlet(Teq), top=Dirichlet(Teq))
params = StefanParams(Tm, rhoL; kappa1=1.0, source1=0.0, thermo_bc=GibbsThomson(sigma; kinetic=0.0))
opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=12, interface_band=3.0)
prob = StefanMonoProblem(grid, bc, params, rep, opts)

solver = build_solver(prob; uω0=Teq, uγ0=Teq, speed0=0.0)

dt = 0.001
nsteps = 12
vmax_hist = Float64[]
r_hist = Float64[]

for _ in 1:nsteps
    step!(solver, dt)
    push!(vmax_hist, maximum(abs.(vec(solver.state.speed_full))))
    push!(r_hist, radius_from_volume(solver.cache.model.Vn1; dim=2))
end

println("Mono 2D stationary circle with Gibbs-Thomson equilibrium")
println("  Tm                = ", Tm)
println("  sigma             = ", sigma)
println("  target TΓ         = ", Teq)
println("  final time        = ", solver.state.t)
println("  max interface |V| = ", maximum(vmax_hist))
println("  radius drift      = ", abs(r_hist[end] - R0))
