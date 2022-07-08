abstract type AbstractControlledMarkovChain end

nb_states(::AbstractControlledMarkovChain) = error("Not implemented.")
initial_distribution(::AbstractControlledMarkovChain) = error("Not implemented.")
transition_matrix(::AbstractControlledMarkovChain, u, args...) = error("Not implemented.")
