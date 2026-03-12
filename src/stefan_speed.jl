function _psi_vectors(Vn::AbstractVector{T}, Vn1::AbstractVector{T}, scheme) where {T}
    θ = PenguinDiffusion._theta_from_scheme(T, scheme)
    psip, psim = PenguinDiffusion._psi_functions(T, θ)
    ψp = Vector{T}(undef, length(Vn))
    ψm = Vector{T}(undef, length(Vn))
    @inbounds for i in eachindex(Vn)
        ψp[i] = convert(T, psip(Vn[i], Vn1[i]))
        ψm[i] = convert(T, psim(Vn[i], Vn1[i]))
    end
    return ψp, ψm
end

function _interface_flux_mono(
    model::PenguinDiffusion.MovingDiffusionModelMono{N,T},
    uω_new::AbstractVector{T},
    uγ_new::AbstractVector{T},
    uω_prev::AbstractVector{T},
    uγ_prev::AbstractVector{T},
    t::T,
    dt::T,
    scheme,
) where {N,T}
    cap = something(model.cap_slab)
    ops = something(model.ops_slab)
    θ = PenguinDiffusion._theta_from_scheme(T, scheme)
    _, _, J, L = PenguinDiffusion._weighted_core_ops(cap, ops, model.D, t + θ * dt, model.coeff_mode)

    ψp, ψm = _psi_vectors(model.Vn, model.Vn1, scheme)
    Tω = ψp .* uω_new .+ ψm .* uω_prev
    Tγ = ψp .* uγ_new .+ ψm .* uγ_prev
    return J * Tω + L * Tγ
end

function _interface_flux_diph(
    model::PenguinDiffusion.MovingDiffusionModelDiph{N,T},
    uω1_new::AbstractVector{T},
    uγ1_new::AbstractVector{T},
    uω2_new::AbstractVector{T},
    uγ2_new::AbstractVector{T},
    uω1_prev::AbstractVector{T},
    uγ1_prev::AbstractVector{T},
    uω2_prev::AbstractVector{T},
    uγ2_prev::AbstractVector{T},
    t::T,
    dt::T,
    scheme,
) where {N,T}
    cap1 = something(model.cap1_slab)
    ops1 = something(model.ops1_slab)
    cap2 = something(model.cap2_slab)
    ops2 = something(model.ops2_slab)
    θ = PenguinDiffusion._theta_from_scheme(T, scheme)

    _, _, J1, L1 = PenguinDiffusion._weighted_core_ops(cap1, ops1, model.D1, t + θ * dt, model.coeff_mode)
    _, _, J2, L2 = PenguinDiffusion._weighted_core_ops(cap2, ops2, model.D2, t + θ * dt, model.coeff_mode)

    ψ1p, ψ1m = _psi_vectors(model.V1n, model.V1n1, scheme)
    ψ2p, ψ2m = _psi_vectors(model.V2n, model.V2n1, scheme)

    Tω1 = ψ1p .* uω1_new .+ ψ1m .* uω1_prev
    Tγ1 = ψ1p .* uγ1_new .+ ψ1m .* uγ1_prev
    Tω2 = ψ2p .* uω2_new .+ ψ2m .* uω2_prev
    Tγ2 = ψ2p .* uγ2_new .+ ψ2m .* uγ2_prev

    return J1 * Tω1 + L1 * Tγ1, J2 * Tω2 + L2 * Tγ2
end

function stefan_speed_mono!(
    v_nodes,
    frozen_mask::BitArray{N},
    cache::MonoMovingDiffusionCache{N,T},
    uω_new::AbstractVector{T},
    uγ_new::AbstractVector{T},
    uω_prev::AbstractVector{T},
    uγ_prev::AbstractVector{T},
    rhoL,
    scheme,
    t::T,
    dt::T;
    tol::T=sqrt(eps(T)),
) where {N,T}
    cap = cache.model.cap_slab
    cap === nothing && throw(ArgumentError("moving slab geometry is not initialized"))

    flux = _interface_flux_mono(cache.model, uω_new, uγ_new, uω_prev, uγ_prev, t, dt, scheme)
    Γ = cap.buf.Γ

    vflat = vec(v_nodes)
    fflat = vec(frozen_mask)
    length(vflat) == length(Γ) || throw(DimensionMismatch("v_nodes must have $(length(Γ)) entries"))

    fill!(vflat, zero(T))
    fill!(fflat, false)

    rhoL_T = convert(T, rhoL)
    has_interface = false
    if N == 1
        # In 1D, use the integrated interface flux to recover a single front speed.
        flux_sum = zero(T)
        @inbounds for i in eachindex(vflat)
            γ = Γ[i]
            if isfinite(γ) && γ > tol
                flux_sum += flux[i]
                has_interface = true
            end
        end
        if has_interface
            v_if = -flux_sum / (rhoL_T * dt)
            @inbounds for i in eachindex(vflat)
                γ = Γ[i]
                if isfinite(γ) && γ > tol
                    vflat[i] = v_if
                    fflat[i] = true
                end
            end
        end
    else
        @inbounds for i in eachindex(vflat)
            γ = Γ[i]
            if isfinite(γ) && γ > tol
                vflat[i] = -flux[i] / (rhoL_T * γ)
                fflat[i] = true
                has_interface = true
            end
        end
    end

    if !has_interface
        @warn "No interface DOFs detected; Stefan speed set to zero for this step"
    end

    return v_nodes, frozen_mask
end

function stefan_speed_diph!(
    v_nodes,
    frozen_mask::BitArray{N},
    cache::DiphMovingDiffusionCache{N,T},
    uω1_new::AbstractVector{T},
    uγ1_new::AbstractVector{T},
    uω2_new::AbstractVector{T},
    uγ2_new::AbstractVector{T},
    uω1_prev::AbstractVector{T},
    uγ1_prev::AbstractVector{T},
    uω2_prev::AbstractVector{T},
    uγ2_prev::AbstractVector{T},
    rhoL,
    scheme,
    t::T,
    dt::T;
    tol::T=sqrt(eps(T)),
) where {N,T}
    cap1 = cache.model.cap1_slab
    cap2 = cache.model.cap2_slab
    cap1 === nothing && throw(ArgumentError("moving slab geometry is not initialized"))
    cap2 === nothing && throw(ArgumentError("moving slab geometry is not initialized"))

    flux1, flux2 = _interface_flux_diph(
        cache.model,
        uω1_new,
        uγ1_new,
        uω2_new,
        uγ2_new,
        uω1_prev,
        uγ1_prev,
        uω2_prev,
        uγ2_prev,
        t,
        dt,
        scheme,
    )

    Γ = cap1.buf.Γ
    vflat = vec(v_nodes)
    fflat = vec(frozen_mask)
    length(vflat) == length(Γ) || throw(DimensionMismatch("v_nodes must have $(length(Γ)) entries"))

    fill!(vflat, zero(T))
    fill!(fflat, false)

    rhoL_T = convert(T, rhoL)
    has_interface = false
    if N == 1
        flux_sum = zero(T)
        @inbounds for i in eachindex(vflat)
            γ = Γ[i]
            if isfinite(γ) && γ > tol
                flux_sum += flux1[i] + flux2[i]
                has_interface = true
            end
        end
        if has_interface
            v_if = -flux_sum / (rhoL_T * dt)
            @inbounds for i in eachindex(vflat)
                γ = Γ[i]
                if isfinite(γ) && γ > tol
                    vflat[i] = v_if
                    fflat[i] = true
                end
            end
        end
    else
        @inbounds for i in eachindex(vflat)
            γ = Γ[i]
            if isfinite(γ) && γ > tol
                # Phase operators use opposite interface normals; physical jump is flux1 + flux2.
                vflat[i] = -(flux1[i] + flux2[i]) / (rhoL_T * γ)
                fflat[i] = true
                has_interface = true
            end
        end
    end

    if !has_interface
        @warn "No interface DOFs detected; Stefan speed set to zero for this step"
    end

    return v_nodes, frozen_mask
end
