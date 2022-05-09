## Structs

"""
    DiscreteMarkovChain

Discrete-time Markov chain with finite state space.

# Fields
- `p0::AbstractVector`: initial state distribution.
- `P::AbstractMatrix`: state transition matrix.
"""
Base.@kwdef struct DiscreteMarkovChain{R1<:Real,R2<:Real}
    p0::Vector{R1}
    P::Matrix{R2}

    function DiscreteMarkovChain{R1,R2}(
        p0::Vector{R1}, P::Matrix{R2}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(p0)
        @assert is_trans_mat(P)
        return new{R1,R2}(p0, P)
    end
end

function DiscreteMarkovChain(p0::Vector{R1}, P::Matrix{R2}) where {R1<:Real,R2<:Real}
    return DiscreteMarkovChain{R1,R2}(p0, P)
end

@inline DensityInterface.DensityKind(::DiscreteMarkovChain) = HasDensity()

"""
    DiscreteMarkovChainPrior

Define a Dirichlet prior on the initial distribution and on the transition matrix of a [`DiscreteMarkovChain`](@ref).

# Fields
- `p0_α::AbstractVector`: Dirichlet parameter for the initial distribution
- `P_α::AbstractMatrix`: Dirichlet parameters for the transition matrix
"""
Base.@kwdef struct DiscreteMarkovChainPrior{R1<:Real,R2<:Real}
    p0_α::Vector{R1}
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
nb_states(mc::DiscreteMarkovChain) = length(mc.p0)

"""
    initial_distribution(mc::DiscreteMarkovChain)

Return the vector of initial state probabilities of `mc`.
"""
initial_distribution(mc::DiscreteMarkovChain) = mc.p0

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
    p_stat = real.(eigvecs(transition_matrix(mc)')[:, end])
    return p_stat / sum(p_stat)
end

"""
    zero_prior(mc::DiscreteMarkovChain)

Build a flat prior, for which MAP is equivalent to MLE.
"""
function flat_prior(mc::DiscreteMarkovChain{R1,R2}) where {R1<:Real, R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    P_α = ones(R2, S, S)
    return DiscreteMarkovChainPrior(p0_α, P_α)
end
