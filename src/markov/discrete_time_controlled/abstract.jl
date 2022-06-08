"""
    AbstractControlledDiscreteMarkovChain

Controlled discrete-time Markov chain with finite state space.
"""
abstract type AbstractControlledDiscreteMarkovChain <: AbstractMarkovChain end

## Access

function transition_matrix(mc::AbstractControlledDiscreteMarkovChain, u)
    return error("not implemented")
end
