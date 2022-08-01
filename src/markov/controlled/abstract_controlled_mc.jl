abstract type AbstractControlledMarkovChain end

nb_states(::AbstractControlledMarkovChain) = error("Not implemented.")
initial_distribution(::AbstractControlledMarkovChain) = error("Not implemented.")
transition_matrix(::AbstractControlledMarkovChain, u, ps, st) = error("Not implemented.")
