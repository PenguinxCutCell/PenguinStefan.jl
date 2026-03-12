function stefan_lambda_neumann(Ste::T; atol::T=T(1e-12), maxiter::Int=200) where {T<:Real}
    Ste > zero(T) || throw(ArgumentError("Ste must be positive"))

    f(λ) = sqrt(T(pi)) * λ * exp(λ^2) * erf(λ) - Ste
    lo = zero(T)
    hi = one(T)
    while f(hi) <= zero(T)
        hi *= T(2)
        hi > T(50) && throw(ArgumentError("failed to bracket lambda for Ste=$Ste"))
    end

    for _ in 1:maxiter
        mid = (lo + hi) / T(2)
        fm = f(mid)
        if abs(fm) < atol || (hi - lo) < atol
            return mid
        end
        if fm > zero(T)
            hi = mid
        else
            lo = mid
        end
    end
    return (lo + hi) / T(2)
end

function neumann_1d_similarity(
    x::T,
    t::T;
    Ts::T=one(T),
    Tm::T=zero(T),
    α::T=one(T),
    Ste::T=one(T),
    λ::Union{Nothing,T}=nothing,
) where {T<:Real}
    λv = λ === nothing ? stefan_lambda_neumann(Ste) : λ
    t <= zero(T) && return (T=Tm, s=zero(T), Vn=zero(T), λ=λv)

    s = 2 * λv * sqrt(α * t)
    if x <= s
        η = x / (2 * sqrt(α * t))
        Tex = Ts - (Ts - Tm) * (erf(η) / erf(λv))
    else
        Tex = Tm
    end
    Vn = λv * sqrt(α / t)
    return (T=Tex, s=s, Vn=Vn, λ=λv)
end

function manufactured_planar_1d(
    x::T,
    t::T;
    s0::T,
    V::T,
    Tm::T=zero(T),
    A::T=one(T),
    kappa::T=one(T),
    rhoL::T=one(T),
) where {T<:Real}
    s = s0 + V * t
    Tex = x <= s ? (Tm + A * (s - x)) : Tm
    source = A * V
    flux = -kappa * A
    vflux = -flux / rhoL
    return (T=Tex, s=s, source=source, flux=flux, vflux=vflux)
end

function frank_disk_exact(
    r::T,
    t::T;
    s0::T,
    Tinf::T=one(T),
    Tm::T=zero(T),
) where {T<:Real}
    t > zero(T) || throw(ArgumentError("t must be positive for Frank disk exact solution"))

    R = s0 * sqrt(t)
    Vn = s0 / (2 * sqrt(t))
    ξ = r / sqrt(t)

    F0 = SpecialFunctions.expint((s0^2) / 4)
    Fξ = SpecialFunctions.expint((ξ^2) / 4)

    Tex = if r <= R
        Tm
    else
        Tinf * (one(T) - Fξ / F0)
    end
    return (T=Tex, R=R, Vn=Vn)
end
