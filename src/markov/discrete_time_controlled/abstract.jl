"""
    AbstractControlledDiscreteMarkovChain

Controlled discrete-time Markov chain with finite state space.
"""
abstract type AbstractControlledDiscreteMarkovChain <: AbstractMarkovChain end

## Access

initial_distribution(mc::AbstractControlledDiscreteMarkovChain) = error("not implemented")
transition_matrix(mc::AbstractControlledDiscreteMarkovChain, u) = error("not implemented")

function transition_probability(mc::AbstractControlledDiscreteMarkovChain, i, j, u)
    return transition_matrix(mc, u)[i, j]
end
