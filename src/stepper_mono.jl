function _eval_state_value(v, x::SVector{N,T}, t::T) where {N,T}
    if v isa Number
        return convert(T, v)
    elseif v isa Function
        if applicable(v, x..., t)
            return convert(T, v(x..., t))
        elseif applicable(v, x...)
            return convert(T, v(x...))
        end
    end
    throw(ArgumentError("state initializer must be scalar, vector, or callback (x...) / (x..., t)"))
end

function _init_component_values(v0, xpts::Vector{SVector{N,T}}, t::T, default_v) where {N,T}
    nt = length(xpts)
    if v0 === nothing
        out = Vector{T}(undef, nt)
        @inbounds for i in 1:nt
            out[i] = _eval_state_value(default_v, xpts[i], t)
        end
        return out
    elseif v0 isa AbstractVector
        length(v0) == nt || throw(DimensionMismatch("initializer length must be $nt"))
        return convert(Vector{T}, v0)
    end

    out = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        out[i] = _eval_state_value(v0, xpts[i], t)
    end
    return out
end

function _init_speed(speed0, ::Type{T}, shp::NTuple{N,Int}) where {N,T}
    if speed0 === nothing
        return zeros(T, shp)
    elseif speed0 isa Number
        return fill(convert(T, speed0), shp)
    elseif speed0 isa AbstractArray
        size(speed0) == shp || throw(DimensionMismatch("speed0 shape must be $(shp)"))
        return convert(Array{T,N}, speed0)
    end
    throw(ArgumentError("speed0 must be nothing, scalar, or array with shape grid.n"))
end

function build_solver(
    prob::StefanMonoProblem{N,T};
    t0::T=zero(T),
    uω0=nothing,
    uγ0=nothing,
    speed0=nothing,
) where {N,T}
    cache = build_mono_cache(prob)

    ϕ0 = phi_values(prob.interface_rep)
    update_slab_field!(cache.slab, ϕ0, ϕ0, t0, t0 + one(T))
    PenguinDiffusion._build_moving_slab!(cache.model, t0, one(T))
    cap = something(cache.model.cap_slab)

    uω = _init_component_values(uω0, cap.C_ω, t0, prob.params.Tm)
    uγ = _init_component_values(uγ0, cap.C_γ, t0, prob.params.Tm)
    speed = _init_speed(speed0, T, size(ϕ0))
    frozen = falses(size(ϕ0))

    logs = Dict{Symbol,Any}(
        :step => 0,
        :times => T[t0],
        :speed_max => T[],
    )
    state = StefanMonoState{N,T,typeof(speed)}(t0, uω, uγ, speed, frozen, logs)
    return StefanMonoSolver{N,T,typeof(prob),typeof(state),typeof(cache)}(prob, state, cache)
end

function step!(
    solver::StefanMonoSolver{N,T},
    dt::T;
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    prob = solver.problem
    state = solver.state
    cache = solver.cache
    rep = prob.interface_rep

    mode = coupling_mode(rep)
    if mode !== :explicit
        if mode === :ghf_newton
            return coupled_step!(solver, dt; method=method, kwargs...)
        end
        throw(ArgumentError("unsupported coupling_mode `$mode` for $(typeof(rep))"))
    end

    t = state.t
    phi_n = phi_values(rep)
    phi_np1_pred = predict_phi(rep, state.speed_full, t, dt)

    uω_new, uγ_new, uprev, _ = solve_moving_diffusion_mono!(
        cache,
        phi_n,
        phi_np1_pred,
        state.speed_full,
        state.uω,
        state.uγ,
        t,
        dt,
        prob;
        method=method,
        kwargs...,
    )

    lay = cache.model.layout.offsets
    uω_prev = Vector{T}(uprev[lay.ω])
    uγ_prev = Vector{T}(uprev[lay.γ])

    v_new = similar(state.speed_full)
    frozen_new = similar(state.frozen_mask)
    stefan_speed_mono!(
        v_new,
        frozen_new,
        cache,
        uω_new,
        uγ_new,
        uω_prev,
        uγ_prev,
        prob.params.rhoL,
        prob.options.scheme,
        t,
        dt,
    )

    extend_speed!(
        rep,
        v_new;
        frozen=frozen_new,
        nb_iters=prob.options.extend_iters,
        cfl=prob.options.extend_cfl,
        interface_band=prob.options.interface_band,
    )
    advance!(rep, v_new, t, dt)

    step_id = get(state.logs, :step, 0) + 1
    state.logs[:step] = step_id
    if prob.options.reinit && prob.options.reinit_every > 0 && (step_id % prob.options.reinit_every == 0)
        reinit!(rep)
    end

    state.t = t + dt
    state.uω .= uω_new
    state.uγ .= uγ_new
    state.speed_full .= v_new
    state.frozen_mask .= frozen_new

    push!(state.logs[:times], state.t)
    push!(state.logs[:speed_max], maximum(abs.(vec(v_new))))

    return state
end

function solve!(
    solver::StefanMonoSolver{N,T},
    tspan::Tuple{T,T};
    dt::T,
    save_history::Bool=true,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    t0, tend = tspan
    tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    state = solver.state
    if abs(state.t - t0) > sqrt(eps(T)) * max(one(T), abs(t0))
        throw(ArgumentError("solver state time $(state.t) must match tspan start $t0"))
    end

    history = Vector{NamedTuple}()
    if save_history
        push!(history, (t=state.t, uω=copy(state.uω), uγ=copy(state.uγ), speed=copy(state.speed_full), phi=phi_values(solver.problem.interface_rep)))
    end

    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    while state.t < tend - tol
        dt_step = min(dt, tend - state.t)
        step!(solver, dt_step; method=method, kwargs...)
        if save_history
            push!(history, (t=state.t, uω=copy(state.uω), uγ=copy(state.uγ), speed=copy(state.speed_full), phi=phi_values(solver.problem.interface_rep)))
        end
    end

    return (
        solver=solver,
        state=solver.state,
        times=copy(solver.state.logs[:times]),
        history=history,
    )
end
