abstract type AbstractInterfaceRep{N,T} end

interface_grid(rep::AbstractInterfaceRep) = throw(MethodError(interface_grid, (rep,)))
phi_values(rep::AbstractInterfaceRep) = throw(MethodError(phi_values, (rep,)))
advance!(rep::AbstractInterfaceRep, v_nodes, t, dt) = throw(MethodError(advance!, (rep, v_nodes, t, dt)))
predict_phi(rep::AbstractInterfaceRep, v_prev_nodes, t, dt) = throw(MethodError(predict_phi, (rep, v_prev_nodes, t, dt)))
coupling_mode(rep::AbstractInterfaceRep) = :explicit

function coupled_step!(solver, dt; kwargs...)
    throw(MethodError(coupled_step!, (solver, dt)))
end

function extend_speed!(rep::AbstractInterfaceRep, v_nodes; kwargs...)
    return v_nodes
end

reinit!(rep::AbstractInterfaceRep) = nothing
