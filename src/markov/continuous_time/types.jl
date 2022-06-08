## Structs

"""
    ContinuousMarkovChain

Continuous-time Markov chain with finite state space.

# Fields
- `p0::Vector`: initial state distribution.
- `Q::Matrix`: state rates matrix.
"""
Base.@kwdef struct ContinuousMarkovChain{R1<:Real,R2<:Real}
    p0::Vector{R1}
    Q::Matrix{R2}

    function ContinuousMarkovChain{R1,R2}(
        p0::Vector{R1}, Q::Matrix{R2}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(p0)
        @assert is_rates_mat(Q)
        return new{R1,R2}(p0, Q)
    end
end

function ContinuousMarkovChain(p0::Vector{R1}, P::Matrix{R2}) where {R1<:Real,R2<:Real}
    return ContinuousMarkovChain{R1,R2}(p0, P)
end

@inline DensityInterface.DensityKind(::ContinuousMarkovChain) = HasDensity()

"""
    ContinuousMarkovChainPrior

Define a Dirichlet prior on the initial distribution and a Gamma prior on the rates matrix of a [`ContinuousMarkovChain`](@ref).

# Fields
- `p0_α::Vector`: Dirichlet parameter for the initial distribution
- `Q_α::Matrix`: Gamma shape parameters for the rates matrix
- `Q_β::Vector`: Gamma rate parameters for the rates matrix
"""
Base.@kwdef struct ContinuousMarkovChainPrior{R1<:Real,R2<:Real,R3<:Real}
    p0_α::Vector{R1}
    Q_α::Matrix{R2}
    Q_β::Vector{R3}
end

@inline DensityInterface.DensityKind(::ContinuousMarkovChainPrior) = HasDensity()

"""
    ContinuousMarkovChainStats

Store sufficient statistics for the likelihood of a [`ContinuousMarkovChain`](@ref) sample.

# Fields
- `initialization_count::Vector`: count initializations in each state
- `transition_count::Matrix`: count transitions between each pair of states
- `duration::Vector`: measure duration in each state
"""
Base.@kwdef struct ContinuousMarkovChainStats{R1<:Real,R2<:Real,R3<:Real}
    initialization_count::Vector{R1}
    transition_count::Matrix{R2}
    duration::Vector{R3}

    function ContinuousMarkovChainStats{R1,R2,R3}(
        initialization_count::Vector{R1}, transition_count::Matrix{R2}, duration::Vector{R3}
    ) where {R1<:Real,R2<:Real,R3<:Real}
        @assert !any(isnan, initialization_count)
        @assert !any(isnan, transition_count)
        @assert !any(isnan, duration)
        return new{R1,R2,R3}(initialization_count, transition_count, duration)
    end
end

function ContinuousMarkovChainStats(
    initialization_count::Vector{R1}, transition_count::Matrix{R2}, duration::Vector{R3}
) where {R1<:Real,R2<:Real,R3<:Real}
    return ContinuousMarkovChainStats{R1,R2,R3}(
        initialization_count, transition_count, duration
    )
end
