mutable struct LevelSetRep{N,T,SG,LG,BC,INT,RC} <: AbstractInterfaceRep{N,T}
    space_grid::SG
    ls_grid::LG
    levelset
    bc::BC
    integrator::INT
    reinit_cfg::RC
    reinit_tau::T
end

function _to_ls_grid(grid::CartesianGrids.CartesianGrid{N,T}) where {N,T}
    return LevelSetMethods.CartesianGrid(Tuple(grid.lc), Tuple(grid.hc), grid.n)
end

@inline function _ls_lerp(a::T, b::T, θ::T) where {T}
    return (one(T) - θ) * a + θ * b
end

function _ls_interp1(xnodes::AbstractVector{T}, vals::AbstractVector{T}, x::T) where {T}
    n = length(xnodes)
    n == 1 && return vals[1]
    if x <= xnodes[1]
        return vals[1]
    elseif x >= xnodes[end]
        return vals[end]
    end
    i = clamp(searchsortedlast(xnodes, x), 1, n - 1)
    x0 = xnodes[i]
    x1 = xnodes[i + 1]
    θ = (x - x0) / (x1 - x0)
    return _ls_lerp(vals[i], vals[i + 1], θ)
end

function _ls_interp2(
    xnodes::AbstractVector{T},
    ynodes::AbstractVector{T},
    vals::AbstractMatrix{T},
    x::T,
    y::T,
) where {T}
    nx = length(xnodes)
    ny = length(ynodes)
    nx == 1 && ny == 1 && return vals[1, 1]

    i = x <= xnodes[1] ? 1 : (x >= xnodes[end] ? nx - 1 : clamp(searchsortedlast(xnodes, x), 1, nx - 1))
    j = y <= ynodes[1] ? 1 : (y >= ynodes[end] ? ny - 1 : clamp(searchsortedlast(ynodes, y), 1, ny - 1))
    x0, x1 = xnodes[i], xnodes[i + 1]
    y0, y1 = ynodes[j], ynodes[j + 1]
    θx = x1 == x0 ? zero(T) : (x - x0) / (x1 - x0)
    θy = y1 == y0 ? zero(T) : (y - y0) / (y1 - y0)

    v00 = vals[i, j]
    v10 = vals[i + 1, j]
    v01 = vals[i, j + 1]
    v11 = vals[i + 1, j + 1]
    vx0 = _ls_lerp(v00, v10, θx)
    vx1 = _ls_lerp(v01, v11, θx)
    return _ls_lerp(vx0, vx1, θy)
end

function _ls_interp3(
    xnodes::AbstractVector{T},
    ynodes::AbstractVector{T},
    znodes::AbstractVector{T},
    vals::Array{T,3},
    x::T,
    y::T,
    z::T,
) where {T}
    nx = length(xnodes)
    ny = length(ynodes)
    nz = length(znodes)
    nx == 1 && ny == 1 && nz == 1 && return vals[1, 1, 1]

    i = x <= xnodes[1] ? 1 : (x >= xnodes[end] ? nx - 1 : clamp(searchsortedlast(xnodes, x), 1, nx - 1))
    j = y <= ynodes[1] ? 1 : (y >= ynodes[end] ? ny - 1 : clamp(searchsortedlast(ynodes, y), 1, ny - 1))
    k = z <= znodes[1] ? 1 : (z >= znodes[end] ? nz - 1 : clamp(searchsortedlast(znodes, z), 1, nz - 1))

    x0, x1 = xnodes[i], xnodes[i + 1]
    y0, y1 = ynodes[j], ynodes[j + 1]
    z0, z1 = znodes[k], znodes[k + 1]
    θx = x1 == x0 ? zero(T) : (x - x0) / (x1 - x0)
    θy = y1 == y0 ? zero(T) : (y - y0) / (y1 - y0)
    θz = z1 == z0 ? zero(T) : (z - z0) / (z1 - z0)

    c000 = vals[i, j, k]
    c100 = vals[i + 1, j, k]
    c010 = vals[i, j + 1, k]
    c110 = vals[i + 1, j + 1, k]
    c001 = vals[i, j, k + 1]
    c101 = vals[i + 1, j, k + 1]
    c011 = vals[i, j + 1, k + 1]
    c111 = vals[i + 1, j + 1, k + 1]

    c00 = _ls_lerp(c000, c100, θx)
    c10 = _ls_lerp(c010, c110, θx)
    c01 = _ls_lerp(c001, c101, θx)
    c11 = _ls_lerp(c011, c111, θx)
    c0 = _ls_lerp(c00, c10, θy)
    c1 = _ls_lerp(c01, c11, θy)
    return _ls_lerp(c0, c1, θz)
end

function _array_levelset_callback(grid::CartesianGrids.CartesianGrid{N,T}, phi0::AbstractArray{T,N}) where {N,T}
    xyz = CartesianGrids.grid1d(grid)
    if N == 1
        return (x) -> _ls_interp1(xyz[1], phi0, x)
    elseif N == 2
        return (x, y) -> _ls_interp2(xyz[1], xyz[2], phi0, x, y)
    elseif N == 3
        return (x, y, z) -> _ls_interp3(xyz[1], xyz[2], xyz[3], phi0, x, y, z)
    end
    throw(ArgumentError("array initialization is supported only in 1D/2D/3D"))
end

function LevelSetRep(
    grid::CartesianGrids.CartesianGrid{N,T},
    phi0;
    bc=LevelSetMethods.NeumannBC(),
    integrator=LevelSetMethods.RK3(),
    reinit_cfg=LevelSetMethods.Reinitializer(),
    reinit_tau::Real=0.0,
) where {N,T}
    ls_grid = _to_ls_grid(grid)
    ls = if phi0 isa Function
        LevelSetMethods.LevelSet(x -> phi0(x...), ls_grid)
    elseif phi0 isa AbstractArray
        size(phi0) == grid.n || throw(DimensionMismatch("phi0 size must match grid.n"))
        cb = _array_levelset_callback(grid, convert(Array{T,N}, phi0))
        LevelSetMethods.LevelSet(x -> cb(x...), ls_grid)
    else
        throw(ArgumentError("phi0 must be a function or an array with shape grid.n"))
    end
    return LevelSetRep{N,T,typeof(grid),typeof(ls_grid),typeof(bc),typeof(integrator),typeof(reinit_cfg)}(
        grid,
        ls_grid,
        ls,
        bc,
        integrator,
        reinit_cfg,
        convert(T, reinit_tau),
    )
end

interface_grid(rep::LevelSetRep) = rep.ls_grid
phi_values(rep::LevelSetRep) = copy(values(rep.levelset))

function _advect_levelset(
    ls,
    grid,
    v_nodes,
    bc,
    integrator,
    t,
    dt,
)
    speed_field = v_nodes isa LevelSetMethods.MeshField ? v_nodes : LevelSetMethods.MeshField(v_nodes, grid, nothing)
    term = LevelSetMethods.NormalMotionTerm(speed=speed_field)
    eq = LevelSetMethods.LevelSetEquation(; terms=(term,), levelset=ls, bc=bc, t=t, integrator=integrator)
    LevelSetMethods.integrate!(eq, t + dt, dt)
    return LevelSetMethods.current_state(eq)
end

function predict_phi(rep::LevelSetRep{N,T}, v_prev_nodes, t::T, dt::T) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    ls_pred = deepcopy(rep.levelset)
    ls_pred = _advect_levelset(ls_pred, rep.ls_grid, v_prev_nodes, rep.bc, rep.integrator, t, dt)
    return copy(values(ls_pred))
end

function advance!(rep::LevelSetRep{N,T}, v_nodes, t::T, dt::T)::Nothing where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    rep.levelset = _advect_levelset(rep.levelset, rep.ls_grid, v_nodes, rep.bc, rep.integrator, t, dt)
    return nothing
end

@inline function _val_neumann(F, i)
    i = clamp(i, firstindex(F), lastindex(F))
    return F[i]
end

function _extend_velocity_normal_1d!(F::AbstractVector{T}, ϕ::AbstractVector{T}, Δ::Real; nb_iters::Int=20, frozen=nothing) where {T<:Real}
    τ = 0.45 * Δ
    Fnew = similar(F)
    n = length(F)
    for _ in 1:nb_iters
        @inbounds for i in 1:n
            if frozen !== nothing && frozen[i]
                Fnew[i] = F[i]
                continue
            end
            im = max(i - 1, 1)
            ip = min(i + 1, n)
            ϕx = (ϕ[ip] - ϕ[im]) / (2Δ)
            mag = abs(ϕx)
            if mag < 1e-14
                Fnew[i] = F[i]
                continue
            end
            nx = ϕx / mag
            S = ϕ[i] / sqrt(ϕ[i]^2 + Δ^2)
            Sn = S * nx
            dF = if Sn > 0
                (F[i] - _val_neumann(F, i - 1)) / Δ
            else
                (_val_neumann(F, i + 1) - F[i]) / Δ
            end
            Fnew[i] = F[i] - τ * (Sn * dF)
        end
        F .= Fnew
    end
    return F
end

function _extend_speed_nearest_1d!(v::AbstractVector{T}, frozen::AbstractVector{Bool}) where {T}
    idx = findall(identity, frozen)
    isempty(idx) && return v

    @inbounds for i in eachindex(v)
        k = searchsortedfirst(idx, i)
        if k <= 1
            v[i] = v[idx[1]]
        elseif k > length(idx)
            v[i] = v[idx[end]]
        else
            il = idx[k - 1]
            ir = idx[k]
            v[i] = (i - il) <= (ir - i) ? v[il] : v[ir]
        end
    end
    return v
end

@inline function _cd_x(ϕ, i, j, nx, Δ)
    im = max(i - 1, 1)
    ip = min(i + 1, nx)
    return (ϕ[ip, j] - ϕ[im, j]) / (2Δ)
end

@inline function _cd_y(ϕ, i, j, ny, Δ)
    jm = max(j - 1, 1)
    jp = min(j + 1, ny)
    return (ϕ[i, jp] - ϕ[i, jm]) / (2Δ)
end

@inline function _val2(F, i, j, nx, ny)
    ii = clamp(i, 1, nx)
    jj = clamp(j, 1, ny)
    return F[ii, jj]
end

function _extend_velocity_normal_2d!(F::AbstractMatrix{T}, ϕ::AbstractMatrix{T}, Δ::Real; nb_iters::Int=20, frozen=nothing) where {T<:Real}
    nx, ny = size(F)
    τ = 0.45 * Δ
    Fnew = similar(F)
    for _ in 1:nb_iters
        @inbounds for j in 1:ny, i in 1:nx
            if frozen !== nothing && frozen[i, j]
                Fnew[i, j] = F[i, j]
                continue
            end
            ϕx = _cd_x(ϕ, i, j, nx, Δ)
            ϕy = _cd_y(ϕ, i, j, ny, Δ)
            mag = sqrt(ϕx^2 + ϕy^2)
            if mag < 1e-14
                Fnew[i, j] = F[i, j]
                continue
            end
            nx_comp = ϕx / mag
            ny_comp = ϕy / mag
            S = ϕ[i, j] / sqrt(ϕ[i, j]^2 + Δ^2)
            Snx = S * nx_comp
            Sny = S * ny_comp

            dFx = if Snx > 0
                (F[i, j] - _val2(F, i - 1, j, nx, ny)) / Δ
            else
                (_val2(F, i + 1, j, nx, ny) - F[i, j]) / Δ
            end
            dFy = if Sny > 0
                (F[i, j] - _val2(F, i, j - 1, nx, ny)) / Δ
            else
                (_val2(F, i, j + 1, nx, ny) - F[i, j]) / Δ
            end

            Fnew[i, j] = F[i, j] - τ * (Snx * dFx + Sny * dFy)
        end
        F .= Fnew
    end
    return F
end

function _seed_opposite_interface_side!(
    v_nodes::AbstractArray{T,N},
    frozen_src::BitArray{N},
    ϕ::AbstractArray{T,N},
) where {N,T}
    frozen_ext = copy(frozen_src)
    any(frozen_src) || return frozen_ext
    acc = zeros(T, size(v_nodes))
    cnt = zeros(Int, size(v_nodes))

    @inbounds for I in CartesianIndices(frozen_src)
        frozen_src[I] || continue
        ϕI = ϕ[I]
        vI = v_nodes[I]
        for d in 1:N
            off = ntuple(k -> (k == d ? 1 : 0), N)
            Jp = I + CartesianIndex(off)
            if checkbounds(Bool, frozen_src, Jp)
                ϕJ = ϕ[Jp]
                if !frozen_src[Jp] && ((ϕI <= zero(T) && ϕJ >= zero(T)) || (ϕI >= zero(T) && ϕJ <= zero(T)))
                    acc[Jp] += vI
                    cnt[Jp] += 1
                end
            end
            Jm = I - CartesianIndex(off)
            if checkbounds(Bool, frozen_src, Jm)
                ϕJ = ϕ[Jm]
                if !frozen_src[Jm] && ((ϕI <= zero(T) && ϕJ >= zero(T)) || (ϕI >= zero(T) && ϕJ <= zero(T)))
                    acc[Jm] += vI
                    cnt[Jm] += 1
                end
            end
        end
    end

    @inbounds for I in CartesianIndices(frozen_src)
        c = cnt[I]
        if c > 0
            v_nodes[I] = acc[I] / c
            frozen_ext[I] = true
        end
    end
    return frozen_ext
end

function extend_speed!(
    rep::LevelSetRep,
    v_nodes;
    frozen,
    nb_iters::Int,
    cfl,
    interface_band,
    min_norm=1e-10,
)
    if ndims(v_nodes) == 1 && !(frozen === nothing) && any(frozen)
        return _extend_speed_nearest_1d!(v_nodes, vec(frozen))
    end

    frozen_use = frozen
    if !(frozen === nothing) && ndims(v_nodes) >= 2
        ϕ = values(rep.levelset)
        frozen_use = _seed_opposite_interface_side!(v_nodes, frozen, ϕ)
    end

    try
        LevelSetMethods.extend_along_normals!(
            v_nodes,
            rep.levelset;
            frozen=frozen_use,
            nb_iters=nb_iters,
            cfl=cfl,
            interface_band=interface_band,
            min_norm=min_norm,
        )
        return v_nodes
    catch
        Δ = minimum(LevelSetMethods.meshsize(rep.ls_grid))
        ϕ = values(rep.levelset)
        if ndims(v_nodes) == 1
            _extend_velocity_normal_1d!(v_nodes, ϕ, Δ; nb_iters=nb_iters, frozen=frozen_use)
        elseif ndims(v_nodes) == 2
            _extend_velocity_normal_2d!(v_nodes, ϕ, Δ; nb_iters=nb_iters, frozen=frozen_use)
        else
            throw(ArgumentError("fallback normal extension supports 1D/2D only"))
        end
        return v_nodes
    end
end

function reinit!(rep::LevelSetRep)::Nothing
    rep.reinit_cfg === nothing && return nothing
    try
        LevelSetMethods.reinitialize!(rep.levelset, rep.reinit_cfg)
        return nothing
    catch
        Δ = minimum(LevelSetMethods.meshsize(rep.ls_grid))
        dτ = 0.3 * Δ
        τf = rep.reinit_tau > 0 ? rep.reinit_tau : 3Δ
        term = LevelSetMethods.ReinitializationTerm()
        eq = LevelSetMethods.LevelSetEquation(; terms=(term,), levelset=rep.levelset, bc=rep.bc, t=0.0, integrator=rep.integrator)
        LevelSetMethods.integrate!(eq, τf, dτ)
        rep.levelset = LevelSetMethods.current_state(eq)
        return nothing
    end
end
