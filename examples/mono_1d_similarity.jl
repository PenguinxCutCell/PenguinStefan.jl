using CartesianGrids
using PenguinBCs
using PenguinStefan

function interface_position(rep::LevelSetRep)
    ϕ = PenguinStefan.phi_values(rep)
    x = CartesianGrids.grid1d(rep.space_grid, 1)
    for i in 1:(length(x) - 1)
        a = ϕ[i]
        b = ϕ[i + 1]
        if a == 0
            return x[i]
        elseif a * b < 0
            θ = abs(a) / (abs(a) + abs(b))
            return (1 - θ) * x[i] + θ * x[i + 1]
        end
    end
    return NaN
end

grid = CartesianGrid((0.0,), (1.0,), (257,))
s0 = 0.247
rep = LevelSetRep(grid, x -> x - s0)

bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0))
params = StefanParams(0.5, 2.0; kappa1=1.0, source1=0.0)
opts = StefanOptions(; scheme=:BE, reinit=true, reinit_every=5, extend_iters=12)
prob = StefanMonoProblem(grid, bc, params, rep, opts)

solver = build_solver(prob; uω0=(x, t) -> 1.0 - 2x, uγ0=0.5)
out = solve!(solver, (0.0, 0.05); dt=0.0025, save_history=false)

println("Mono 1D Stefan run")
println("  final time: ", out.state.t)
println("  interface position ~ ", interface_position(rep))
println("  max speed: ", maximum(abs.(vec(out.state.speed_full))))
