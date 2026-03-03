using CartesianGrids
using PenguinBCs
using PenguinStefan

grid = CartesianGrid((0.0,), (1.0,), (193,))
rep = LevelSetRep(grid, x -> x - 0.53)

bc = BorderConditions(; left=Dirichlet(0.5), right=Dirichlet(0.5))
params = StefanParams(0.5, 1.0; kappa1=1.0, kappa2=2.0, source1=0.0, source2=0.0)
opts = StefanOptions(; scheme=:BE, reinit=false, extend_iters=8)
prob = StefanDiphProblem(grid, bc, params, rep, opts)

solver = build_solver(prob; uω10=0.5, uγ10=0.5, uω20=0.5, uγ20=0.5)
out = solve!(solver, (0.0, 0.03); dt=0.003, save_history=false)

println("Diph 1D stationary")
println("  final time: ", out.state.t)
println("  max speed: ", maximum(abs.(vec(out.state.speed_full))))
println("  frozen nodes: ", count(out.state.frozen_mask))
