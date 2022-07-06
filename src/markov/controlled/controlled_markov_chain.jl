struct ControlledMarkovChain{R<:Real,M} <: AbstractMarkovChain
    p0::Vector{R}
    P_model::M
end

## Access

nb_states(mc::ControlledMarkovChain) = length(mc.p0)
initial_distribution(mc::ControlledMarkovChain) = mc.p0
transition_matrix(mc::ControlledMarkovChain, u) = mc.P_model(u)
