abstract type AbstractControlledMarkovChain end

nb_states(::AbstractControlledMarkovChain) = error("Not implemented.")
initial_distribution(::AbstractControlledMarkovChain) = error("Not implemented.")

function transition_matrix(::AbstractControlledMarkovChain, control, params)
    return error("Not implemented.")
end

function transition_matrix!(
    ::AbstractMatrix, ::AbstractControlledMarkovChain, control, params
)
    return error("Not implemented.")
end
