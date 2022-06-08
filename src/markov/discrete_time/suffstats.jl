"""
    DiscreteMarkovChainStats

Store sufficient statistics for the likelihood of a [`DiscreteMarkovChain`](@ref) sample.

# Fields
- `initialization_count::Vector`: count initializations in each state
- `transition_count::Matrix`: count transitions between each pair of states
"""
struct DiscreteMarkovChainStats{R1<:Real,R2<:Real}
    initialization_count::Vector{R1}
    transition_count::Matrix{R2}

    function DiscreteMarkovChainStats{R1,R2}(
        initialization_count::AbstractVector{<:Real},
        transition_count::AbstractMatrix{<:Real},
    ) where {R1<:Real,R2<:Real}
        @assert !any(isnan, initialization_count)
        @assert !any(isnan, transition_count)
        return new{R1,R2}(
            convert(Vector{R1}, initialization_count), convert(Matrix{R2}, transition_count)
        )
    end
end

function DiscreteMarkovChainStats(
    initialization_count::Vector{R1}, transition_count::Matrix{R2}
) where {R1<:Real,R2<:Real}
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

## Compute sufficient stats

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}}, initialization_count, transition_count
) where {R1,R2}
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}}, state_sequence::AbstractVector{<:Integer}
) where {R1,R2}
    S, T = maximum(state_sequence), length(state_sequence)
    initialization_count = zeros(R1, S)
    transition_count = zeros(R2, S, S)
    initialization_count[first(state_sequence)] = one(R1)
    for t in 1:(T - 1)
        i, j = state_sequence[t], state_sequence[t + 1]
        transition_count[i, j] += one(R2)
    end
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}},
    state_sequences::AbstractVector{<:AbstractVector{<:Integer}},
) where {R1,R2}
    S = mapreduce(maximum, max, state_sequences)
    initialization_count = zeros(R1, S)
    transition_count = zeros(R2, S, S)
    for state_sequence in state_sequences
        initialization_count[first(state_sequence)] += one(R1)
        T = length(state_sequence)
        for t in 1:(T - 1)
            i, j = state_sequence[t], state_sequence[t + 1]
            transition_count[i, j] += one(R2)
        end
    end
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end
