## Structs

"""
    DiscreteMarkovChain

Discrete-time Markov chain with finite state space.

# Fields
- `π0::AbstractVector`: initial state distribution.
- `P::AbstractMatrix`: state transition matrix.
"""
Base.@kwdef struct DiscreteMarkovChain{
    R1<:Real,R2<:Real,V1<:AbstractVector{R1},M2<:AbstractMatrix{R2}
}
    π0::V1
    P::M2
end

@inline DensityInterface.DensityKind(::DiscreteMarkovChain) = HasDensity()

"""
    DiscreteMarkovChainPrior

Define a Dirichlet prior on the initial distribution and on the transitions from each state.

# Fields
- `π0_α::AbstractVector`: Dirichlet parameter for the initial distribution
- `P_α::AbstractMatrix`: Dirichlet parameters for the transition matrix
"""
Base.@kwdef struct DiscreteMarkovChainPrior{
    R1<:Real,R2<:Real,V1<:AbstractVector{R1},M2<:AbstractMatrix{R2}
}
    π0_α::V1
    P_α::M2
end

@inline DensityInterface.DensityKind(::DiscreteMarkovChainPrior) = HasDensity()

"""
    DiscreteMarkovChainStats

Sufficient statistics for the likelihood of a `DiscreteMarkovChain`.

# Fields
- `initialization_count::AbstractVector`: count initializations in each state
- `transition_count::AbstractMatrix`: count transitions between each pair of states
"""
Base.@kwdef struct DiscreteMarkovChainStats{
    R1<:Real,R2<:Real,V1<:AbstractVector{R1},M2<:AbstractMatrix{R2}
}
    initialization_count::V1
    transition_count::M2
end

## Access

nb_states(mc::DiscreteMarkovChain) = length(mc.π0)
initial_distribution(mc::DiscreteMarkovChain) = mc.π0
transition_matrix(mc::DiscreteMarkovChain) = mc.P

function stationary_distribution(mc::DiscreteMarkovChain)
    π = real.(eigvecs(transition_matrix(mc)')[:, end])
    return π / sum(π)
end
