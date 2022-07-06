"""
    AbstractMarkovChain

Abstract supertype for discrete-time Markov chains with finite state space.
"""
abstract type AbstractMarkovChain end

@inline DensityInterface.DensityKind(::AbstractMarkovChain) = HasDensity()

function nb_states end
function initial_distribution end
function transition_matrix end
