using CartesianGrids
using PenguinBCs
using PenguinStefan

grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (121, 121))
rep = LevelSetRep(grid, (x, y) -> 0.35 - sqrt(x^2 + y^2))

bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(1.0), bottom=Dirichlet(1.0), top=Dirichlet(1.0))
params = StefanParams(0.0, 1.0; kappa1=1.0, source1=0.0)
opts = StefanOptions(; scheme=:CN, reinit=true, reinit_every=4, extend_iters=16, interface_band=4.0)
prob = StefanMonoProblem(grid, bc, params, rep, opts)

solver = build_solver(prob; uω0=0.0, uγ0=0.0, speed0=0.0)
out = solve!(solver, (0.0, 0.03); dt=0.0015, save_history=false)

ϕ = PenguinStefan.phi_values(rep)
Δ = minimum(CartesianGrids.step.(CartesianGrids.grid1d(grid)))
interface_nodes = count(abs.(ϕ) .<= (1.5 * Δ))

println("Mono 2D circle melt")
println("  final time: ", out.state.t)
println("  max speed: ", maximum(abs.(vec(out.state.speed_full))))
println("  interface-band node count: ", interface_nodes)
