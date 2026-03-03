function build_solver(
    prob::StefanDiphProblem{N,T};
    t0::T=zero(T),
    uω10=nothing,
    uγ10=nothing,
    uω20=nothing,
    uγ20=nothing,
    speed0=nothing,
) where {N,T}
    cache = build_diph_cache(prob)

    ϕ0 = phi_values(prob.interface_rep)
    update_slab_field!(cache.slab, ϕ0, ϕ0, t0, t0 + one(T))
    PenguinDiffusion._build_moving_slab!(cache.model, t0, one(T))
    cap1 = something(cache.model.cap1_slab)
    cap2 = something(cache.model.cap2_slab)

    uω1 = _init_component_values(uω10, cap1.C_ω, t0, prob.params.Tm)
    uγ1 = _init_component_values(uγ10, cap1.C_γ, t0, prob.params.Tm)
    uω2 = _init_component_values(uω20, cap2.C_ω, t0, prob.params.Tm)
    uγ2 = _init_component_values(uγ20, cap2.C_γ, t0, prob.params.Tm)

    speed = _init_speed(speed0, T, size(ϕ0))
    frozen = falses(size(ϕ0))

    logs = Dict{Symbol,Any}(
        :step => 0,
        :times => T[t0],
        :speed_max => T[],
    )
    state = StefanDiphState{N,T,typeof(speed)}(t0, uω1, uγ1, uω2, uγ2, speed, frozen, logs)
    return StefanDiphSolver{N,T,typeof(prob),typeof(state),typeof(cache)}(prob, state, cache)
end

function step!(
    solver::StefanDiphSolver{N,T},
    dt::T;
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    prob = solver.problem
    state = solver.state
    cache = solver.cache
    rep = prob.interface_rep

    t = state.t
    phi_n = phi_values(rep)
    phi_np1_pred = predict_phi(rep, state.speed_full, t, dt)

    uω1_new, uγ1_new, uω2_new, uγ2_new, uprev, _ = solve_moving_diffusion_diph_stefan!(
        cache,
        phi_n,
        phi_np1_pred,
        state.uω1,
        state.uγ1,
        state.uω2,
        state.uγ2,
        t,
        dt,
        prob;
        method=method,
        kwargs...,
    )

    lay = cache.model.layout.offsets
    uω1_prev = Vector{T}(uprev[lay.ω1])
    uγ1_prev = Vector{T}(uprev[lay.γ1])
    uω2_prev = Vector{T}(uprev[lay.ω2])
    uγ2_prev = Vector{T}(uprev[lay.γ2])

    v_new = similar(state.speed_full)
    frozen_new = similar(state.frozen_mask)
    stefan_speed_diph!(
        v_new,
        frozen_new,
        cache,
        uω1_new,
        uγ1_new,
        uω2_new,
        uγ2_new,
        uω1_prev,
        uγ1_prev,
        uω2_prev,
        uγ2_prev,
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
    state.uω1 .= uω1_new
    state.uγ1 .= uγ1_new
    state.uω2 .= uω2_new
    state.uγ2 .= uγ2_new
    state.speed_full .= v_new
    state.frozen_mask .= frozen_new

    push!(state.logs[:times], state.t)
    push!(state.logs[:speed_max], maximum(abs.(vec(v_new))))

    return state
end

function solve!(
    solver::StefanDiphSolver{N,T},
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
        push!(history, (
            t=state.t,
            uω1=copy(state.uω1), uγ1=copy(state.uγ1),
            uω2=copy(state.uω2), uγ2=copy(state.uγ2),
            speed=copy(state.speed_full),
            phi=phi_values(solver.problem.interface_rep),
        ))
    end

    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    while state.t < tend - tol
        dt_step = min(dt, tend - state.t)
        step!(solver, dt_step; method=method, kwargs...)
        if save_history
            push!(history, (
                t=state.t,
                uω1=copy(state.uω1), uγ1=copy(state.uγ1),
                uω2=copy(state.uω2), uγ2=copy(state.uγ2),
                speed=copy(state.speed_full),
                phi=phi_values(solver.problem.interface_rep),
            ))
        end
    end

    return (
        solver=solver,
        state=solver.state,
        times=copy(solver.state.logs[:times]),
        history=history,
    )
end
