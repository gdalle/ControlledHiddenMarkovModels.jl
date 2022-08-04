abstract type AbstractControlledMarkovChain end

nb_states(mc::AbstractControlledMarkovChain) = error("Not implemented.")

## Log versions

function log_initial_distribution(mc::AbstractControlledMarkovChain, parameters)
    return error("Not implemented.")
end

function log_transition_matrix!(
    logP::AbstractMatrix, mc::AbstractControlledMarkovChain, control, parameters
)
    return error("Not implemented.")
end

function log_transition_matrix(mc::AbstractControlledMarkovChain, control, parameters)
    return error("Not implemented.")
end

## Normal versions

function initial_distribution(mc::AbstractControlledMarkovChain, parameters)
    return exp.(log_initial_distribution(mc, parameters))
end

function transition_matrix!(
    P::AbstractMatrix, mc::AbstractControlledMarkovChain, control, parameters
)
    log_transition_matrix!(P, mc, control, parameters)
    P .= exp.(P)
    return P
end

function transition_matrix(mc::AbstractControlledMarkovChain, control, parameters)
    return exp.(log_transition_matrix(mc, control, parameters))
end
