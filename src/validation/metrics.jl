function active_mask(cap::AssembledCapacity{N,T}; tol::T=zero(T)) where {N,T}
    m = falses(cap.ntotal)
    @inbounds for i in eachindex(m)
        v = cap.buf.V[i]
        m[i] = isfinite(v) && v > tol
    end
    return m
end

function phase_volume(cap::AssembledCapacity{N,T}; tol::T=zero(T)) where {N,T}
    s = zero(T)
    @inbounds for i in 1:cap.ntotal
        v = cap.buf.V[i]
        if isfinite(v) && v > tol
            s += v
        end
    end
    return s
end

function phase_volume(V::AbstractVector{T}; tol::T=zero(T)) where {T<:Real}
    s = zero(T)
    @inbounds for i in eachindex(V)
        v = V[i]
        if isfinite(v) && v > tol
            s += v
        end
    end
    return s
end

function box_measure(cap::AssembledCapacity{N,T}) where {N,T}
    m = one(T)
    @inbounds for d in 1:N
        m *= cap.xyz[d][end] - cap.xyz[d][1]
    end
    return m
end

function interface_position_from_volume(cap::AssembledCapacity{1,T}; tol::T=zero(T)) where {T}
    return phase_volume(cap; tol=tol)
end

function interface_position_from_volume(V::AbstractVector{T}; tol::T=zero(T)) where {T<:Real}
    return phase_volume(V; tol=tol)
end

function radius_from_volume(cap::AssembledCapacity{N,T}; complement::Bool=false, tol::T=zero(T)) where {N,T}
    N in (2, 3) || throw(ArgumentError("radius_from_volume is defined for N=2 or N=3"))
    v = phase_volume(cap; tol=tol)
    if complement
        v = max(box_measure(cap) - v, zero(T))
    end
    if N == 2
        return sqrt(v / T(pi))
    end
    return cbrt(3v / (4T(pi)))
end

function radius_from_volume(
    V::AbstractVector{T};
    dim::Int=2,
    complement::Bool=false,
    domain_measure::Union{Nothing,T}=nothing,
    tol::T=zero(T),
) where {T<:Real}
    dim in (2, 3) || throw(ArgumentError("radius_from_volume is defined for dim=2 or dim=3"))
    v = phase_volume(V; tol=tol)
    if complement
        domain_measure === nothing && throw(ArgumentError("domain_measure is required when complement=true"))
        v = max(domain_measure - v, zero(T))
    end
    if dim == 2
        return sqrt(v / T(pi))
    end
    return cbrt(3v / (4T(pi)))
end

function _eval_exact(Texact, x::SVector{N,T}, t::T) where {N,T}
    if applicable(Texact, x..., t)
        return convert(T, Texact(x..., t))
    elseif applicable(Texact, x...)
        return convert(T, Texact(x...))
    end
    throw(ArgumentError("exact solution callback must accept (x...) or (x..., t)"))
end

function temperature_error_norms(
    uω::AbstractVector{T},
    Texact,
    cap::AssembledCapacity{N,T},
    t::T;
    tol::T=zero(T),
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {N,T}
    length(uω) == cap.ntotal || throw(DimensionMismatch("uω length must be $(cap.ntotal)"))
    if !(weights === nothing)
        length(weights) == cap.ntotal || throw(DimensionMismatch("weights length must be $(cap.ntotal)"))
    end

    l1 = zero(T)
    l2 = zero(T)
    linf = zero(T)
    wsum = zero(T)

    @inbounds for i in 1:cap.ntotal
        w = weights === nothing ? cap.buf.V[i] : weights[i]
        if !(isfinite(w) && w > tol)
            continue
        end
        e = abs(uω[i] - _eval_exact(Texact, cap.C_ω[i], t))
        l1 += w * e
        l2 += w * e^2
        linf = max(linf, e)
        wsum += w
    end
    return (L1=l1, L2=sqrt(l2), Linf=linf, weight=wsum)
end

function stefan_residual_metrics(
    Vn::AbstractVector{T},
    Vn1::AbstractVector{T},
    Γ::AbstractVector{T},
    vflux::AbstractVector{T};
    tol::T=sqrt(eps(T)),
) where {T}
    n = length(Vn)
    length(Vn1) == n || throw(DimensionMismatch("Vn1 length mismatch"))
    length(Γ) == n || throw(DimensionMismatch("Γ length mismatch"))
    length(vflux) == n || throw(DimensionMismatch("vflux length mismatch"))

    r = zeros(T, n)
    maxr = zero(T)
    maxv = zero(T)
    l2 = zero(T)
    m = 0

    @inbounds for i in 1:n
        γ = Γ[i]
        if !(isfinite(γ) && γ > tol)
            continue
        end
        ri = (Vn1[i] - Vn[i]) / γ - vflux[i]
        r[i] = ri
        maxr = max(maxr, abs(ri))
        maxv = max(maxv, abs(vflux[i]))
        l2 += ri^2
        m += 1
    end

    l2 = m > 0 ? sqrt(l2 / m) : zero(T)
    rel = maxv > zero(T) ? maxr / maxv : maxr
    return (maxabs=maxr, l2=l2, relmax=rel, residual=r, count=m)
end

function fit_order(hs::AbstractVector{T}, errs::AbstractVector{T}) where {T<:Real}
    length(hs) == length(errs) || throw(DimensionMismatch("hs and errs lengths must match"))
    n = length(hs)
    n >= 2 || throw(ArgumentError("need at least two levels to fit order"))

    pairwise = Vector{T}(undef, n - 1)
    @inbounds for k in 1:(n - 1)
        pairwise[k] = log(errs[k] / errs[k + 1]) / log(hs[k] / hs[k + 1])
    end

    x = log.(hs)
    y = log.(errs)
    xm = sum(x) / n
    ym = sum(y) / n
    num = zero(T)
    den = zero(T)
    @inbounds for k in 1:n
        dx = x[k] - xm
        num += dx * (y[k] - ym)
        den += dx^2
    end
    slope = den > zero(T) ? num / den : zero(T)
    return (pairwise=pairwise, order_global=slope)
end
