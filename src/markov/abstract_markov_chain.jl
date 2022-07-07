"""
    AbstractMarkovChain

Abstract supertype for discrete-time Markov chains with finite state space.
"""
abstract type AbstractMarkovChain end

@inline DensityInterface.DensityKind(::AbstractMarkovChain) = HasDensity()

nb_states(::AbstractMarkovChain) = error("not implemented")
initial_distribution(::AbstractMarkovChain) = error("not implemented")
transition_matrix(::AbstractMarkovChain) = error("not implemented")
