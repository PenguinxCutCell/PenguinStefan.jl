mutable struct GlobalHFRep{N,T,AT} <: AbstractInterfaceRep{N,T}
    grid::CartesianGrids.CartesianGrid{N,T}
    axis::Int
    phi::Array{T,N}
    xf::AT
    periodic_transverse::Bool
    interp::Symbol
    xf_prev::AT
    dt_prev::T
end

function _normalize_ghf_interp(interp::Symbol)
    if interp === :linear || interp === :cubic
        return interp
    elseif interp === :pwlinear
        return :linear
    elseif interp === :cubic_spline
        return :cubic
    end
    throw(ArgumentError("unsupported interpolation `$interp`; expected :linear/:cubic"))
end

function _ghf_phi_from_initializer(phi0::AbstractArray, grid::CartesianGrids.CartesianGrid{N,T}) where {N,T}
    size(phi0) == grid.n || throw(DimensionMismatch("phi0 size must match grid.n"))
    return convert(Array{T,N}, phi0)
end

function _ghf_phi_from_initializer(phi0::Function, grid::CartesianGrids.CartesianGrid{N,T}) where {N,T}
    xyz = ntuple(d -> collect(T, CartesianGrids.grid1d(grid, d)), N)
    phi = Array{T,N}(undef, grid.n...)
    @inbounds for I in CartesianIndices(phi)
        x = ntuple(d -> xyz[d][I[d]], N)
        phi[I] = convert(T, phi0(x...))
    end
    return phi
end

function GlobalHFRep(
    grid::CartesianGrids.CartesianGrid{N,T},
    phi0;
    axis::Union{Int,Symbol,Val}=1,
    periodic_transverse::Bool=false,
    interp::Symbol=:linear,
    method::Symbol=:zero_crossing,
) where {N,T}
    axis_idx = GlobalHeightFunctions.axis_to_index(axis, Val(N))
    interp_mode = _normalize_ghf_interp(interp)
    phi = _ghf_phi_from_initializer(phi0, grid)
    xf = GlobalHeightFunctions.xf_from_sdf(phi, grid; axis=axis_idx, method=method)
    periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf)
    xf_prev = copy(xf)
    return GlobalHFRep{N,T,typeof(xf)}(grid, axis_idx, phi, xf, periodic_transverse, interp_mode, xf_prev, zero(T))
end

interface_grid(rep::GlobalHFRep) = rep.grid
phi_values(rep::GlobalHFRep) = copy(rep.phi)
coupling_mode(::GlobalHFRep) = :ghf_newton

function predict_phi(rep::GlobalHFRep{N,T}, speed_prev, t::T, dt::T) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    xf_guess = copy(rep.xf)

    if rep.dt_prev > zero(T)
        ratio = dt / rep.dt_prev
        xf_guess .= rep.xf .+ ratio .* (rep.xf .- rep.xf_prev)
    end

    rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_guess)

    return GlobalHeightFunctions.phi_from_xf(
        xf_guess,
        rep.grid;
        axis=rep.axis,
        interp=rep.interp,
        periodic=rep.periodic_transverse,
    )
end

function advance!(rep::GlobalHFRep, v_nodes, t, dt)
    error("GlobalHFRep is advanced by coupled iteration, not by advance!")
end

extend_speed!(rep::GlobalHFRep, v_nodes; kwargs...) = v_nodes
reinit!(rep::GlobalHFRep) = nothing
