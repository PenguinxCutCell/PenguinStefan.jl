mutable struct DiphMovingDiffusionCache{N,T,SF,MD,SY}
    slab::SF
    model::MD
    sys::SY
end

function build_diph_cache(prob::StefanDiphProblem{N,T}) where {N,T}
    ϕ = phi_values(prob.interface_rep)
    xyz = ntuple(d -> collect(T, CartesianGrids.grid1d(prob.grid, d)), N)
    slab = SlabPhiField{N,T,typeof(copy(ϕ))}(xyz, copy(ϕ), copy(ϕ), zero(T), one(T))

    model = PenguinDiffusion.MovingDiffusionModelDiph(
        prob.grid,
        slab,
        prob.params.kappa1,
        prob.params.kappa2;
        source=(prob.params.source1, prob.params.source2),
        bc_border=prob.bc_border,
        ic=nothing,
        coeff_mode=:harmonic,
        geom_method=:vofijul,
    )

    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    return DiphMovingDiffusionCache{N,T,typeof(slab),typeof(model),typeof(sys)}(slab, model, sys)
end

function _as_prev_full_diph(
    model,
    uω1_prev::AbstractVector{T},
    uγ1_prev::AbstractVector{T},
    uω2_prev::AbstractVector{T},
    uγ2_prev::AbstractVector{T},
) where {T}
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    nt = length(lay.ω1)

    length(uω1_prev) == nt || throw(DimensionMismatch("uω1_prev length must be $nt"))
    length(uγ1_prev) == nt || throw(DimensionMismatch("uγ1_prev length must be $nt"))
    length(uω2_prev) == nt || throw(DimensionMismatch("uω2_prev length must be $nt"))
    length(uγ2_prev) == nt || throw(DimensionMismatch("uγ2_prev length must be $nt"))

    u = zeros(T, nsys)
    u[lay.ω1] .= uω1_prev
    u[lay.γ1] .= uγ1_prev
    u[lay.ω2] .= uω2_prev
    u[lay.γ2] .= uγ2_prev
    return u
end

function assemble_moving_diffusion_diph_stefan!(
    cache::DiphMovingDiffusionCache{N,T},
    phi_n,
    phi_np1,
    speed_full_prev,
    uω1_prev::AbstractVector{T},
    uγ1_prev::AbstractVector{T},
    uω2_prev::AbstractVector{T},
    uγ2_prev::AbstractVector{T},
    t::T,
    dt::T,
    prob::StefanDiphProblem{N,T},
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    update_slab_field!(cache.slab, phi_n, phi_np1, t, t + dt)
    uprev = _as_prev_full_diph(cache.model, uω1_prev, uγ1_prev, uω2_prev, uγ2_prev)

    θ = PenguinDiffusion._theta_from_scheme(T, prob.options.scheme)
    psip, psim = PenguinDiffusion._psi_functions(T, θ)

    PenguinDiffusion._build_moving_slab!(cache.model, t, dt)
    cap1 = something(cache.model.cap1_slab)
    ops1 = something(cache.model.ops1_slab)
    cap2 = something(cache.model.cap2_slab)
    ops2 = something(cache.model.ops2_slab)

    nt = cap1.ntotal
    lay = cache.model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    K1, C1, J1, L1 = PenguinDiffusion._weighted_core_ops(cap1, ops1, cache.model.D1, t + θ * dt, cache.model.coeff_mode)
    K2, C2, J2, L2 = PenguinDiffusion._weighted_core_ops(cap2, ops2, cache.model.D2, t + θ * dt, cache.model.coeff_mode)

    f1_n, f2_n = PenguinDiffusion._source_values_diph(cap1, cache.model.source1, cap2, cache.model.source2, t)
    f1_n1, f2_n1 = PenguinDiffusion._source_values_diph(cap1, cache.model.source1, cap2, cache.model.source2, t + dt)

    M1n = spdiagm(0 => cache.model.V1n)
    M1n1 = spdiagm(0 => cache.model.V1n1)
    M2n = spdiagm(0 => cache.model.V2n)
    M2n1 = spdiagm(0 => cache.model.V2n1)

    ψ1p = T[psip(cache.model.V1n[i], cache.model.V1n1[i]) for i in 1:nt]
    ψ1m = T[psim(cache.model.V1n[i], cache.model.V1n1[i]) for i in 1:nt]
    ψ2p = T[psip(cache.model.V2n[i], cache.model.V2n1[i]) for i in 1:nt]
    ψ2m = T[psim(cache.model.V2n[i], cache.model.V2n1[i]) for i in 1:nt]
    Ψ1p = spdiagm(0 => ψ1p)
    Ψ1m = spdiagm(0 => ψ1m)
    Ψ2p = spdiagm(0 => ψ2p)
    Ψ2m = spdiagm(0 => ψ2m)

    A11 = M1n1 + θ * (K1 * Ψ1p)
    A12 = -(M1n1 - M1n) + θ * (C1 * Ψ1p)
    A33 = M2n1 + θ * (K2 * Ψ2p)
    A34 = -(M2n1 - M2n) + θ * (C2 * Ψ2p)

    IΓ1 = cap1.Γ
    IΓ2 = cap2.Γ

    Z = spzeros(T, nt, nt)
    A21 = Z
    A22 = IΓ1
    A23 = Z
    A24 = Z
    A41 = Z
    A42 = Z
    A43 = Z
    A44 = IΓ2

    uω1 = Vector{T}(uprev[lay.ω1])
    uγ1 = Vector{T}(uprev[lay.γ1])
    uω2 = Vector{T}(uprev[lay.ω2])
    uγ2 = Vector{T}(uprev[lay.γ2])

    bω1 = (M1n - (one(T) - θ) * (K1 * Ψ1m)) * uω1
    bω1 .-= (one(T) - θ) .* ((C1 * Ψ1m) * uγ1)
    bω1 .+= θ .* (cap1.V * f1_n1) .+ (one(T) - θ) .* (cap1.V * f1_n)

    bω2 = (M2n - (one(T) - θ) * (K2 * Ψ2m)) * uω2
    bω2 .-= (one(T) - θ) .* ((C2 * Ψ2m) * uγ2)
    bω2 .+= θ .* (cap2.V * f2_n1) .+ (one(T) - θ) .* (cap2.V * f2_n)

    tg1 = _gibbs_thomson_trace_values(prob, phi_np1, speed_full_prev, cap1.C_γ, t + dt)
    tg2 = _gibbs_thomson_trace_values(prob, phi_np1, speed_full_prev, cap2.C_γ, t + dt)
    bγ1 = IΓ1 * tg1
    bγ2 = IΓ2 * tg2

    A, b = if PenguinDiffusion._is_canonical_diph_layout(lay, nt)
        (
            [A11 A12 Z Z;
             A21 A22 A23 A24;
             Z Z A33 A34;
             A41 A42 A43 A44],
            vcat(bω1, bγ1, bω2, bγ2),
        )
    else
        Awork = spzeros(T, nsys, nsys)
        bwork = zeros(T, nsys)
        PenguinDiffusion._insert_block!(Awork, lay.ω1, lay.ω1, A11)
        PenguinDiffusion._insert_block!(Awork, lay.ω1, lay.γ1, A12)
        PenguinDiffusion._insert_block!(Awork, lay.γ1, lay.ω1, A21)
        PenguinDiffusion._insert_block!(Awork, lay.γ1, lay.γ1, A22)
        PenguinDiffusion._insert_block!(Awork, lay.γ1, lay.ω2, A23)
        PenguinDiffusion._insert_block!(Awork, lay.γ1, lay.γ2, A24)

        PenguinDiffusion._insert_block!(Awork, lay.ω2, lay.ω2, A33)
        PenguinDiffusion._insert_block!(Awork, lay.ω2, lay.γ2, A34)
        PenguinDiffusion._insert_block!(Awork, lay.γ2, lay.ω1, A41)
        PenguinDiffusion._insert_block!(Awork, lay.γ2, lay.γ1, A42)
        PenguinDiffusion._insert_block!(Awork, lay.γ2, lay.ω2, A43)
        PenguinDiffusion._insert_block!(Awork, lay.γ2, lay.γ2, A44)

        PenguinDiffusion._insert_vec!(bwork, lay.ω1, bω1)
        PenguinDiffusion._insert_vec!(bwork, lay.γ1, bγ1)
        PenguinDiffusion._insert_vec!(bwork, lay.ω2, bω2)
        PenguinDiffusion._insert_vec!(bwork, lay.γ2, bγ2)
        (Awork, bwork)
    end

    cache.sys.A = A
    cache.sys.b = b
    length(cache.sys.x) == nsys || (cache.sys.x = zeros(T, nsys))
    cache.sys.cache = nothing

    layω1 = UnknownLayout(nt, (ω=lay.ω1,))
    layω2 = UnknownLayout(nt, (ω=lay.ω2,))
    apply_box_bc_mono!(cache.sys.A, cache.sys.b, cap1, ops1, cache.model.D1, cache.model.bc_border; t=t + θ * dt, layout=layω1)
    apply_box_bc_mono!(cache.sys.A, cache.sys.b, cap2, ops2, cache.model.D2, cache.model.bc_border; t=t + θ * dt, layout=layω2)

    active_rows = PenguinDiffusion._diph_row_activity(cap1, cap2, lay)
    cache.sys.A, cache.sys.b = PenguinDiffusion._apply_row_identity_constraints!(cache.sys.A, cache.sys.b, active_rows)
    return cache.sys, uprev
end

function solve_moving_diffusion_diph_stefan!(
    cache::DiphMovingDiffusionCache{N,T},
    phi_n,
    phi_np1,
    speed_full_prev,
    uω1_prev::AbstractVector{T},
    uγ1_prev::AbstractVector{T},
    uω2_prev::AbstractVector{T},
    uγ2_prev::AbstractVector{T},
    t::T,
    dt::T,
    prob::StefanDiphProblem{N,T};
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    sys, uprev = assemble_moving_diffusion_diph_stefan!(
        cache,
        phi_n,
        phi_np1,
        speed_full_prev,
        uω1_prev,
        uγ1_prev,
        uω2_prev,
        uγ2_prev,
        t,
        dt,
        prob,
    )
    solve!(sys; method=method, reuse_factorization=false, kwargs...)
    lay = cache.model.layout.offsets
    uω1_new = Vector{T}(sys.x[lay.ω1])
    uγ1_new = Vector{T}(sys.x[lay.γ1])
    uω2_new = Vector{T}(sys.x[lay.ω2])
    uγ2_new = Vector{T}(sys.x[lay.γ2])
    return uω1_new, uγ1_new, uω2_new, uγ2_new, uprev, sys
end
