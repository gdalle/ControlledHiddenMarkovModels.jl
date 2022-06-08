"""
    AbstractMarkovChain

Abstract supertype for Markov chains with finite state space.
"""
abstract type AbstractMarkovChain end

abstract type AbstractMarkovChainPrior end

@inline DensityInterface.DensityKind(::AbstractMarkovChain) = HasDensity()
@inline DensityInterface.DensityKind(::AbstractMarkovChainPrior) = HasDensity()

## Interface

"""
    initial_distribution(mc::AbstractMarkovChain)

Return the vector of initial state probabilities of `mc`.
"""
initial_distribution(mc::AbstractMarkovChain) = error("not implemented")

"""
    initial_probability(mc::AbstractMarkovChain, i)

Return the probability of `i` being the initial state of `mc`.
"""
initial_probability(mc::AbstractMarkovChain, i::Integer) = initial_distribution(mc)[i]

"""
    stationary_distribution(mc::AbstractMarkovChain)

Compute the equilibrium distribution of `mc` using its eigendecomposition.
"""
stationary_distribution(mc::AbstractMarkovChain) = error("not implemented")

"""
    nb_states(mc::AbstractMarkovChain)

Return the number of states of `mc`.
"""
nb_states(mc::AbstractMarkovChain) = length(initial_distribution(mc))

"""
    flat_prior(mc::AbstractMarkovChain)

Build a flat prior for `mc`, under which MAP is equivalent to MLE.
"""
flat_prior(mc::AbstractMarkovChain) = error("not implemented")
