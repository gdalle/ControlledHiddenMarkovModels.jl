abstract type AbstractControlledMarkovChain end

nb_states(mc::AbstractControlledMarkovChain) = error("Not implemented.")
initial_distribution(mc::AbstractControlledMarkovChain) = error("Not implemented.")

function transition_matrix(mc::AbstractControlledMarkovChain, control, parameters)
    return error("Not implemented.")
end

function transition_matrix!(
    P::AbstractMatrix, mc::AbstractControlledMarkovChain, control, parameters
)
    P .= transition_matrix(mc, control, parameters)
    return P
end
