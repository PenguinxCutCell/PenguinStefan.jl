@inline function _finite_point(x::SVector)
    return all(isfinite, x)
end

function _node_values(v, grid::CartesianGrids.CartesianGrid{N,T}, t::T) where {N,T}
    xyz = CartesianGrids.grid1d(grid)
    out = Array{T,N}(undef, grid.n...)
    @inbounds for I in CartesianIndices(out)
        x = SVector{N,T}(ntuple(d -> convert(T, xyz[d][I[d]]), N))
        out[I] = convert(T, eval_bc(v, x, t))
    end
    return out
end

function _point_values(v, xpts::AbstractVector{<:SVector{N,T}}, t::T) where {N,T}
    out = Vector{T}(undef, length(xpts))
    @inbounds for i in eachindex(xpts)
        x = xpts[i]
        out[i] = _finite_point(x) ? convert(T, eval_bc(v, x, t)) : zero(T)
    end
    return out
end

function _sample_speed_at_interface(speed_full, grid::CartesianGrids.CartesianGrid{N,T}, Cg::AbstractVector{<:SVector{N,T}}) where {N,T}
    out = Vector{T}(undef, length(Cg))
    @inbounds for i in eachindex(Cg)
        x = Cg[i]
        out[i] = _finite_point(x) ? CartesianGeometry.sample_cartesian_field(speed_full, grid, x) : zero(T)
    end
    return out
end

function _sample_curvature_at_interface(phi_nodes, grid::CartesianGrids.CartesianGrid{N,T}, Cg::AbstractVector{<:SVector{N,T}}) where {N,T}
    kappa_nodes = CartesianGeometry.levelset_curvature(phi_nodes, grid)
    out = Vector{T}(undef, length(Cg))
    @inbounds for i in eachindex(Cg)
        x = Cg[i]
        out[i] = _finite_point(x) ? CartesianGeometry.sample_cartesian_field(kappa_nodes, grid, x) : zero(T)
    end
    return out
end

function _gibbs_thomson_trace_values(
    prob::Union{StefanMonoProblem{N,T},StefanDiphProblem{N,T}},
    phi_nodes,
    speed_full,
    Cg::AbstractVector{<:SVector{N,T}},
    t::T,
) where {N,T}
    Tm_g = _point_values(prob.params.Tm, Cg, t)

    if prob.params.thermo_bc === nothing
        return Tm_g
    end

    gt = prob.params.thermo_bc
    gt isa GibbsThomson || throw(ArgumentError("thermo_bc must be nothing or PenguinBCs.GibbsThomson"))

    kappa_g = _sample_curvature_at_interface(phi_nodes, prob.grid, Cg)
    v_g = _sample_speed_at_interface(speed_full, prob.grid, Cg)
    sigma_g = _point_values(gt.capillary, Cg, t)
    mu_g = _point_values(gt.kinetic, Cg, t)

    return Tm_g .- sigma_g .* kappa_g .- mu_g .* v_g
end

function _mono_interface_trace_callback(
    prob::StefanMonoProblem{N,T},
    phi_nodes,
    speed_full_prev,
    ttrace::T,
) where {N,T}
    if prob.params.thermo_bc === nothing
        return prob.params.Tm
    end

    gt = prob.params.thermo_bc
    gt isa GibbsThomson || throw(ArgumentError("thermo_bc must be nothing or PenguinBCs.GibbsThomson"))
    kappa_nodes = CartesianGeometry.levelset_curvature(phi_nodes, prob.grid)
    grid = prob.grid
    Tm = prob.params.Tm

    return let grid=grid, kappa_nodes=kappa_nodes, speed_full_prev=speed_full_prev, ttrace=ttrace, Tm=Tm, gt=gt
        function (x...)
            xx = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
            Tmval = convert(T, eval_bc(Tm, xx, ttrace))
            sigma = convert(T, eval_bc(gt.capillary, xx, ttrace))
            mu = convert(T, eval_bc(gt.kinetic, xx, ttrace))
            kappa = CartesianGeometry.sample_cartesian_field(kappa_nodes, grid, xx)
            v = CartesianGeometry.sample_cartesian_field(speed_full_prev, grid, xx)
            return Tmval - sigma * kappa - mu * v
        end
    end
end
