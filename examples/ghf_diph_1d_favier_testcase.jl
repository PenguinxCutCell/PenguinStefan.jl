using CartesianGrids
using PenguinBCs
using PenguinStefan

# Favier-style two-phase Stefan test case (1D, GHF):
# - Domain [0,1], boundary temperatures: T(0)=1, T(1)=0
# - Initial interface: s0 = 1/5
# - Melting temperature: Tm = 1/5
# - Initial profile: (exp(beta*(x-1)) - 1) / (exp(-beta) - 1), beta=8.041
# - Stefan number: 1 (rhoL = 1 here)
# - Expected steady state: T = 1 - x, s = 4/5

Tm = 0.2
rhoL = 1.0
k1 = 1.0
k2 = 1.0
s0 = 0.2
s_ss = 0.8
beta = 8.041

t0 = 0.0
tf = 0.6

function T0(x)
    return (exp(beta * (x - 1)) - 1) / (exp(-beta) - 1)
end

Tsteady(x) = 1 - x

grid = CartesianGrid((0.0,), (1.0,), (257,))
rep = GlobalHFRep(grid, x -> x - s0; axis=:x, interp=:linear)

bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(0.0))
params = StefanParams(Tm, rhoL; kappa1=k1, kappa2=k2, source1=0.0, source2=0.0)
opts = StefanOptions(
    ;
    scheme=:CN,
    reinit=false,
    coupling_max_iter=25,
    coupling_tol=1e-8,
    coupling_reltol=1e-8,
    coupling_damping=1.0,
)
prob = StefanDiphProblem(grid, bc, params, rep, opts)

solver = build_solver(
    prob;
    t0=t0,
    uω10=(x, t) -> T0(x),
    uγ10=Tm,
    uω20=(x, t) -> T0(x),
    uγ20=Tm,
    speed0=0.0,
)

solve!(solver, (t0, tf); dt=0.002, save_history=false)

cap1 = solver.cache.model.cap1_slab
cap2 = solver.cache.model.cap2_slab
terr1 = temperature_error_norms(solver.state.uω1, (x, t) -> Tsteady(x), cap1, tf)
terr2 = temperature_error_norms(solver.state.uω2, (x, t) -> Tsteady(x), cap2, tf)
s_num = rep.xf[]

println("GHF 1D diphasic Favier test case")
println("beta = $beta, Tm = $Tm, rhoL = $rhoL")
println("final time       = ", solver.state.t)
println("interface s(tf)  = ", s_num)
println("|s(tf)-0.8|      = ", abs(s_num - s_ss))
println("L2(T1-(1-x))     = ", terr1.L2)
println("L2(T2-(1-x))     = ", terr2.L2)
println("max |speed|      = ", maximum(abs.(vec(solver.state.speed_full))))
