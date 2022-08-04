abstract type AbstractControlledMarkovChain end

nb_states(mc::AbstractControlledMarkovChain) = error("Not implemented.")

log_initial_distribution(mc::AbstractControlledMarkovChain) = error("Not implemented.")

initial_distribution(mc::AbstractControlledMarkovChain) = exp.(log_initial_distribution(mc))

function log_transition_matrix(mc::AbstractControlledMarkovChain, control, parameters)
    return error("Not implemented.")
end

function log_transition_matrix!(
    logP::AbstractMatrix, mc::AbstractControlledMarkovChain, control, parameters
)
    logP .= log_transition_matrix(mc, control, parameters)
    return logP
end

function transition_matrix(mc::AbstractControlledMarkovChain, control, parameters)
    return exp.(log_transition_matrix(mc, control, parameters))
end
