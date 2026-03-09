struct StefanParams{T,K1,K2,S1,S2}
    Tm::T
    rhoL::T
    kappa1::K1
    kappa2::K2
    source1::S1
    source2::S2
end

function StefanParams(
    Tm::T,
    rhoL::T;
    kappa1=one(T),
    kappa2=one(T),
    source1=((args...) -> zero(T)),
    source2=((args...) -> zero(T)),
) where {T<:Real}
    return StefanParams{T,typeof(kappa1),typeof(kappa2),typeof(source1),typeof(source2)}(
        Tm,
        rhoL,
        kappa1,
        kappa2,
        source1,
        source2,
    )
end

struct StefanOptions{T,LB,LI,ALG}
    scheme::Symbol
    ls_bc::LB
    ls_integrator::LI
    reinit::Bool
    reinit_every::Int
    extend_iters::Int
    extend_cfl::T
    interface_band::T
    coupling_max_iter::Int
    coupling_tol::T
    coupling_reltol::T
    coupling_damping::T
    alg::ALG
end

function StefanOptions(
    ;
    scheme::Symbol=:BE,
    ls_bc=LevelSetMethods.NeumannBC(),
    ls_integrator=LevelSetMethods.RK3(),
    reinit::Bool=true,
    reinit_every::Int=1,
    extend_iters::Int=20,
    extend_cfl::Real=0.45,
    interface_band::Real=3.0,
    coupling_max_iter::Int=20,
    coupling_tol::Real=1e-10,
    coupling_reltol::Real=1e-10,
    coupling_damping::Real=0.5,
    alg=LinearSolve.KLUFactorization(),
)
    T = promote_type(
        typeof(float(extend_cfl)),
        typeof(float(interface_band)),
        typeof(float(coupling_tol)),
        typeof(float(coupling_reltol)),
        typeof(float(coupling_damping)),
    )
    if !(scheme === :BE || scheme === :CN)
        throw(ArgumentError("scheme must be :BE or :CN"))
    end
    return StefanOptions{T,typeof(ls_bc),typeof(ls_integrator),typeof(alg)}(
        scheme,
        ls_bc,
        ls_integrator,
        reinit,
        reinit_every,
        extend_iters,
        convert(T, extend_cfl),
        convert(T, interface_band),
        coupling_max_iter,
        convert(T, coupling_tol),
        convert(T, coupling_reltol),
        convert(T, coupling_damping),
        alg,
    )
end

struct StefanMonoProblem{N,T,IR,OP,PA}
    grid::CartesianGrids.CartesianGrid{N,T}
    bc_border::BorderConditions
    params::PA
    interface_rep::IR
    options::OP
end

struct StefanDiphProblem{N,T,IR,OP,PA}
    grid::CartesianGrids.CartesianGrid{N,T}
    bc_border::BorderConditions
    params::PA
    interface_rep::IR
    options::OP
end

mutable struct StefanMonoState{N,T,A}
    t::T
    uω::Vector{T}
    uγ::Vector{T}
    speed_full::A
    frozen_mask::BitArray{N}
    logs::Dict{Symbol,Any}
end

mutable struct StefanDiphState{N,T,A}
    t::T
    uω1::Vector{T}
    uγ1::Vector{T}
    uω2::Vector{T}
    uγ2::Vector{T}
    speed_full::A
    frozen_mask::BitArray{N}
    logs::Dict{Symbol,Any}
end

mutable struct StefanMonoSolver{N,T,PR,ST,CA}
    problem::PR
    state::ST
    cache::CA
end

mutable struct StefanDiphSolver{N,T,PR,ST,CA}
    problem::PR
    state::ST
    cache::CA
end

function StefanMonoProblem(
    grid::CartesianGrids.CartesianGrid{N,T},
    bc_border::BorderConditions,
    params::StefanParams,
    interface_rep::AbstractInterfaceRep{N,T},
    options::StefanOptions=StefanOptions(),
) where {N,T}
    return StefanMonoProblem{N,T,typeof(interface_rep),typeof(options),typeof(params)}(
        grid,
        bc_border,
        params,
        interface_rep,
        options,
    )
end

function StefanDiphProblem(
    grid::CartesianGrids.CartesianGrid{N,T},
    bc_border::BorderConditions,
    params::StefanParams,
    interface_rep::AbstractInterfaceRep{N,T},
    options::StefanOptions=StefanOptions(),
) where {N,T}
    return StefanDiphProblem{N,T,typeof(interface_rep),typeof(options),typeof(params)}(
        grid,
        bc_border,
        params,
        interface_rep,
        options,
    )
end
