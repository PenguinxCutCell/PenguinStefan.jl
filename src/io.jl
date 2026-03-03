function state_summary(state::StefanMonoState)
    return (
        t=state.t,
        max_speed=maximum(abs.(vec(state.speed_full))),
        n_frozen=count(state.frozen_mask),
    )
end

function state_summary(state::StefanDiphState)
    return (
        t=state.t,
        max_speed=maximum(abs.(vec(state.speed_full))),
        n_frozen=count(state.frozen_mask),
    )
end
