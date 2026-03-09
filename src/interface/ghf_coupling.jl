function _replace_nonfinite!(A::AbstractArray{T}) where {T}
    @inbounds for i in eachindex(A)
        isfinite(A[i]) || (A[i] = zero(T))
    end
    return A
end

@inline _norm_inf(v::Number) = abs(v)
@inline _norm_inf(v) = maximum(abs, v)

function _height_scale_x0(grid::CartesianGrids.CartesianGrid{N,T}, axis::Int) where {N,T}
    Δ = CartesianGrids.meshsize(grid)
    scale = one(T)
    @inbounds for d in 1:N
        d == axis && continue
        scale *= convert(T, Δ[d])
    end
    x0 = convert(T, first(CartesianGrids.grid1d(grid, axis)))
    return x0, scale
end

function _heights_from_positions(xf, grid::CartesianGrids.CartesianGrid{N,T}, axis::Int) where {N,T}
    x0, scale = _height_scale_x0(grid, axis)
    return scale .* (xf .- x0)
end

function _update_coupling_logs!(logs::Dict{Symbol,Any}, iters::Int, err, err_rel, residual_history)
    if !haskey(logs, :coupling_iters)
        logs[:coupling_iters] = Int[]
    end
    if !haskey(logs, :coupling_residual_inf)
        logs[:coupling_residual_inf] = typeof(err)[]
    end
    if !haskey(logs, :coupling_residual_rel)
        logs[:coupling_residual_rel] = typeof(err_rel)[]
    end
    if !haskey(logs, :coupling_residual_history)
        logs[:coupling_residual_history] = Vector{typeof(copy(residual_history))}()
    end

    push!(logs[:coupling_iters], iters)
    push!(logs[:coupling_residual_inf], err)
    push!(logs[:coupling_residual_rel], err_rel)
    push!(logs[:coupling_residual_history], copy(residual_history))
    logs[:coupling_iters_last] = iters
    logs[:coupling_residual_inf_last] = err
    logs[:coupling_residual_rel_last] = err_rel
    logs[:coupling_residuals_last] = copy(residual_history)
    return logs
end

function _step_mono_ghf!(
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
    rep isa GlobalHFRep{N,T} || throw(ArgumentError("_step_mono_ghf! requires GlobalHFRep"))

    max_iter = prob.options.coupling_max_iter
    tol = prob.options.coupling_tol
    reltol = prob.options.coupling_reltol
    damping = prob.options.coupling_damping

    t = state.t
    phi_n = copy(rep.phi)
    phi_guess = predict_phi(rep, state.speed_full, t, dt)
    xf_guess = GlobalHeightFunctions.xf_from_sdf(phi_guess, prob.grid; axis=rep.axis)
    rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_guess)
    Hguess = _heights_from_positions(xf_guess, prob.grid, rep.axis)

    xf_new = copy(rep.xf)
    uω_new = copy(state.uω)
    uγ_new = copy(state.uγ)
    uω_prev_last = copy(state.uω)
    uγ_prev_last = copy(state.uγ)

    err = typemax(T)
    err_rel = typemax(T)
    iter_done = 0
    residual_history = T[]

    for k in 1:max_iter
        iter_done = k

        uω_k, uγ_k, uprev, _ = solve_moving_diffusion_mono!(
            cache,
            phi_n,
            phi_guess,
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

        flux = _interface_flux_mono(
            cache.model,
            uω_k,
            uγ_k,
            uω_prev,
            uγ_prev,
            t,
            dt,
            prob.options.scheme,
        )

        Vn = _replace_nonfinite!(reshape(copy(cache.model.Vn), prob.grid.n...))
        Vn1 = _replace_nonfinite!(reshape(copy(cache.model.Vn1), prob.grid.n...))
        F = _replace_nonfinite!(reshape(copy(flux), prob.grid.n...))

        Hn_col = GlobalHeightFunctions.column_profile(Vn; axis=rep.axis)
        Hn1_col = GlobalHeightFunctions.column_profile(Vn1; axis=rep.axis)
        Fcol = GlobalHeightFunctions.column_profile(F; axis=rep.axis)

        Rcol = (Hn1_col .- Hn_col) .+ Fcol ./ prob.params.rhoL

        err = _norm_inf(Rcol)
        href = max(_norm_inf(Hguess), eps(T))
        err_rel = err / href
        push!(residual_history, err)

        uω_new .= uω_k
        uγ_new .= uγ_k
        uω_prev_last .= uω_prev
        uγ_prev_last .= uγ_prev

        if err <= tol || err_rel <= reltol
            xf_iter = GlobalHeightFunctions.positions_from_heights(Hguess, prob.grid; axis=rep.axis)
            rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_iter)
            phi_guess .= GlobalHeightFunctions.phi_from_xf(
                xf_iter,
                prob.grid;
                axis=rep.axis,
                interp=rep.interp,
                periodic=rep.periodic_transverse,
            )
            xf_new .= xf_iter
            break
        end

        if Hguess isa Number
            Hguess = Hguess - damping * Rcol
        else
            Hguess .-= damping .* Rcol
        end
        xf_iter = GlobalHeightFunctions.positions_from_heights(Hguess, prob.grid; axis=rep.axis)
        rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_iter)
        phi_guess .= GlobalHeightFunctions.phi_from_xf(
            xf_iter,
            prob.grid;
            axis=rep.axis,
            interp=rep.interp,
            periodic=rep.periodic_transverse,
        )
        xf_new .= xf_iter
    end

    v_new = similar(state.speed_full)
    frozen_new = similar(state.frozen_mask)
    stefan_speed_mono!(
        v_new,
        frozen_new,
        cache,
        uω_new,
        uγ_new,
        uω_prev_last,
        uγ_prev_last,
        prob.params.rhoL,
        prob.options.scheme,
        t,
        dt,
    )

    rep.xf_prev .= rep.xf
    rep.xf .= xf_new
    rep.dt_prev = dt
    rep.phi .= phi_guess

    step_id = get(state.logs, :step, 0) + 1
    state.logs[:step] = step_id
    _update_coupling_logs!(state.logs, iter_done, err, err_rel, residual_history)

    if prob.options.reinit && prob.options.reinit_every > 0 && (step_id % prob.options.reinit_every == 0)
        reinit!(rep)
    end

    state.t = t + dt
    state.uω .= uω_new
    state.uγ .= uγ_new
    state.speed_full .= v_new
    state.frozen_mask .= frozen_new

    push!(state.logs[:times], state.t)
    push!(state.logs[:speed_max], maximum(abs, v_new))

    return state
end

function _step_diph_ghf!(
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
    rep isa GlobalHFRep{N,T} || throw(ArgumentError("_step_diph_ghf! requires GlobalHFRep"))

    max_iter = prob.options.coupling_max_iter
    tol = prob.options.coupling_tol
    reltol = prob.options.coupling_reltol
    damping = prob.options.coupling_damping

    t = state.t
    phi_n = copy(rep.phi)
    phi_guess = predict_phi(rep, state.speed_full, t, dt)
    xf_guess = GlobalHeightFunctions.xf_from_sdf(phi_guess, prob.grid; axis=rep.axis)
    rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_guess)
    Hguess = _heights_from_positions(xf_guess, prob.grid, rep.axis)

    xf_new = copy(rep.xf)
    uω1_new = copy(state.uω1)
    uγ1_new = copy(state.uγ1)
    uω2_new = copy(state.uω2)
    uγ2_new = copy(state.uγ2)
    uω1_prev_last = copy(state.uω1)
    uγ1_prev_last = copy(state.uγ1)
    uω2_prev_last = copy(state.uω2)
    uγ2_prev_last = copy(state.uγ2)

    err = typemax(T)
    err_rel = typemax(T)
    iter_done = 0
    residual_history = T[]

    for k in 1:max_iter
        iter_done = k

        uω1_k, uγ1_k, uω2_k, uγ2_k, uprev, _ = solve_moving_diffusion_diph_stefan!(
            cache,
            phi_n,
            phi_guess,
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

        flux1, flux2 = _interface_flux_diph(
            cache.model,
            uω1_k,
            uγ1_k,
            uω2_k,
            uγ2_k,
            uω1_prev,
            uγ1_prev,
            uω2_prev,
            uγ2_prev,
            t,
            dt,
            prob.options.scheme,
        )

        V1n = _replace_nonfinite!(reshape(copy(cache.model.V1n), prob.grid.n...))
        V1n1 = _replace_nonfinite!(reshape(copy(cache.model.V1n1), prob.grid.n...))
        F = _replace_nonfinite!(reshape(copy(flux1 .+ flux2), prob.grid.n...))

        Hn_col = GlobalHeightFunctions.column_profile(V1n; axis=rep.axis)
        Hn1_col = GlobalHeightFunctions.column_profile(V1n1; axis=rep.axis)
        Fcol = GlobalHeightFunctions.column_profile(F; axis=rep.axis)

        Rcol = (Hn1_col .- Hn_col) .+ Fcol ./ prob.params.rhoL

        err = _norm_inf(Rcol)
        href = max(_norm_inf(Hguess), eps(T))
        err_rel = err / href
        push!(residual_history, err)

        uω1_new .= uω1_k
        uγ1_new .= uγ1_k
        uω2_new .= uω2_k
        uγ2_new .= uγ2_k
        uω1_prev_last .= uω1_prev
        uγ1_prev_last .= uγ1_prev
        uω2_prev_last .= uω2_prev
        uγ2_prev_last .= uγ2_prev

        if err <= tol || err_rel <= reltol
            xf_iter = GlobalHeightFunctions.positions_from_heights(Hguess, prob.grid; axis=rep.axis)
            rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_iter)
            phi_guess .= GlobalHeightFunctions.phi_from_xf(
                xf_iter,
                prob.grid;
                axis=rep.axis,
                interp=rep.interp,
                periodic=rep.periodic_transverse,
            )
            xf_new .= xf_iter
            break
        end

        if Hguess isa Number
            Hguess = Hguess - damping * Rcol
        else
            Hguess .-= damping .* Rcol
        end
        xf_iter = GlobalHeightFunctions.positions_from_heights(Hguess, prob.grid; axis=rep.axis)
        rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_iter)
        phi_guess .= GlobalHeightFunctions.phi_from_xf(
            xf_iter,
            prob.grid;
            axis=rep.axis,
            interp=rep.interp,
            periodic=rep.periodic_transverse,
        )
        xf_new .= xf_iter
    end

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
        uω1_prev_last,
        uγ1_prev_last,
        uω2_prev_last,
        uγ2_prev_last,
        prob.params.rhoL,
        prob.options.scheme,
        t,
        dt,
    )

    rep.xf_prev .= rep.xf
    rep.xf .= xf_new
    rep.dt_prev = dt
    rep.phi .= phi_guess

    step_id = get(state.logs, :step, 0) + 1
    state.logs[:step] = step_id
    _update_coupling_logs!(state.logs, iter_done, err, err_rel, residual_history)

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
    push!(state.logs[:speed_max], maximum(abs, v_new))

    return state
end

function coupled_step!(solver::StefanMonoSolver{N,T}, dt::T; method::Symbol=:direct, kwargs...) where {N,T}
    rep = solver.problem.interface_rep
    rep isa GlobalHFRep || throw(MethodError(coupled_step!, (solver, dt)))
    return _step_mono_ghf!(solver, dt; method=method, kwargs...)
end

function coupled_step!(solver::StefanDiphSolver{N,T}, dt::T; method::Symbol=:direct, kwargs...) where {N,T}
    rep = solver.problem.interface_rep
    rep isa GlobalHFRep || throw(MethodError(coupled_step!, (solver, dt)))
    return _step_diph_ghf!(solver, dt; method=method, kwargs...)
end
