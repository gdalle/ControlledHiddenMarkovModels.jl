abstract type AbstractControlledMarkovChain end

nb_states(::AbstractControlledMarkovChain) = error("not implemented")
initial_distribution(::AbstractControlledMarkovChain) = error("not implemented")
transition_matrix(::AbstractControlledMarkovChain, u, args...) = error("not implemented")
