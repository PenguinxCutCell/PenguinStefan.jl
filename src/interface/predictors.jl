function _zero_speed_like(rep::AbstractInterfaceRep{N,T}) where {N,T}
    ϕ = phi_values(rep)
    return zeros(T, size(ϕ))
end

function predict_phi(rep::AbstractInterfaceRep{N,T}, ::Nothing, t::T, dt::T) where {N,T}
    return predict_phi(rep, _zero_speed_like(rep), t, dt)
end
