"""
    AbstractControlledDiscreteMarkovChain

Controlled discrete-time Markov chain with finite state space.
"""
abstract type AbstractControlledDiscreteMarkovChain <: AbstractDiscreteMarkovChain end

## Access

initial_distribution(mc::AbstractControlledDiscreteMarkovChain) = error("not implemented")

function transition_matrix(mc::AbstractControlledDiscreteMarkovChain, u, ps, st)
    return error("not implemented")
end

function transition_probability(
    mc::AbstractControlledDiscreteMarkovChain, i::Integer, j::Integer, u, ps, st
)
    return transition_matrix(mc, u, ps, st)[i, j]
end
