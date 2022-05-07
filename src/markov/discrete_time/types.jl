## Structs

"""
    DiscreteMarkovChain

Discrete-time Markov chain with finite state space.

# Fields
- `π0::AbstractVector`: initial state distribution.
- `P::AbstractMatrix`: state transition matrix.
"""
Base.@kwdef struct DiscreteMarkovChain{R1<:Real,R2<:Real}
    π0::Vector{R1}
    P::Matrix{R2}

    function DiscreteMarkovChain{R1,R2}(
        π0::Vector{R1}, P::Matrix{R2}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(π0)
        @assert is_trans_mat(P)
        return new{R1,R2}(π0, P)
    end
end

function DiscreteMarkovChain(π0::Vector{R1}, P::Matrix{R2}) where {R1<:Real,R2<:Real}
    return DiscreteMarkovChain{R1,R2}(π0, P)
end

@inline DensityInterface.DensityKind(::DiscreteMarkovChain) = HasDensity()

"""
    DiscreteMarkovChainPrior

Define a Dirichlet prior on the initial distribution and on the transition matrix of a [`DiscreteMarkovChain`](@ref).

# Fields
- `π0_α::AbstractVector`: Dirichlet parameter for the initial distribution
- `P_α::AbstractMatrix`: Dirichlet parameters for the transition matrix
"""
Base.@kwdef struct DiscreteMarkovChainPrior{R1<:Real,R2<:Real}
    π0_α::Vector{R1}
    P_α::Matrix{R2}
end

@inline DensityInterface.DensityKind(::DiscreteMarkovChainPrior) = HasDensity()

"""
    DiscreteMarkovChainStats

Store sufficient statistics for the likelihood of a [`DiscreteMarkovChain`](@ref) sample.

# Fields
- `initialization_count::AbstractVector`: count initializations in each state
- `transition_count::AbstractMatrix`: count transitions between each pair of states
"""
Base.@kwdef struct DiscreteMarkovChainStats{R1<:Real,R2<:Real}
    initialization_count::Vector{R1}
    transition_count::Matrix{R2}

    function DiscreteMarkovChainStats{R1,R2}(
        initialization_count::Vector{R1}, transition_count::Matrix{R2}
    ) where {R1<:Real,R2<:Real}
        @assert !any(isnan, initialization_count)
        @assert !any(isnan, transition_count)
        return new{R1,R2}(initialization_count, transition_count)
    end
end

function DiscreteMarkovChainStats(
    initialization_count::Vector{R1}, transition_count::Matrix{R2}
) where {R1<:Real,R2<:Real}
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

## Access

"""
    nb_states(mc::DiscreteMarkovChain)

Return the number of states of `mc`.
"""
nb_states(mc::DiscreteMarkovChain) = length(mc.π0)

"""
    initial_distribution(mc::DiscreteMarkovChain)

Return the vector of initial state probabilities of `mc`.
"""
initial_distribution(mc::DiscreteMarkovChain) = mc.π0

"""
    transition_matrix(mc::DiscreteMarkovChain)

Return the matrix of transition probabilities of `mc`.
"""
transition_matrix(mc::DiscreteMarkovChain) = mc.P

"""
    stationary_distribution(mc::DiscreteMarkovChain)

Compute the equilibrium distribution of `mc` using its eigendecomposition.
"""
function stationary_distribution(mc::DiscreteMarkovChain)
    π = real.(eigvecs(transition_matrix(mc)')[:, end])
    return π / sum(π)
end

function zero_prior(mc::DiscreteMarkovChain{R1,R2}) where {R1<:Real, R2<:Real}
    S = nb_states(mc)
    π0_α = ones(R1, S)
    P_α = ones(R2, S, S)
    return DiscreteMarkovChainPrior(π0_α, P_α)
end
