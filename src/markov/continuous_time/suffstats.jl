"""
    ContinuousMarkovChainStats

Store sufficient statistics for the likelihood of a [`ContinuousMarkovChain`](@ref) sample.

# Fields
- `initialization_count::Vector`: count initializations in each state
- `transition_count::Matrix`: count transitions between each pair of states
- `duration::Vector`: measure duration in each state
"""
struct ContinuousMarkovChainStats{R1<:Real,R2<:Real,R3<:Real}
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

## Compute sufficient stats

function Distributions.suffstats(
    ::Type{ContinuousMarkovChain{R1,R2}},
    initialization_count::AbstractVector{<:Real},
    transition_count::AbstractMatrix{<:Real},
    duration::AbstractVector{<:Real},
) where {R1,R2}
    return ContinuousMarkovChainStats{R1,R2,R2}(
        Vector{R1}(initialization_count), Matrix{R2}(transition_count), Vector{R2}(duration)
    )
end

function Distributions.suffstats(
    ::Type{ContinuousMarkovChain{R1,R2}}, h::History{<:Integer}
) where {R1,R2}
    state_sequence = event_marks(h)
    transition_times = event_times(h)
    S = maximum(state_sequence)
    initialization_count = zeros(R1, S)
    transition_count = zeros(R2, S, S)
    duration = zeros(R2, S)
    initialization_count[first(state_sequence)] = one(R1)
    for t in 1:(nb_events(h) - 1)
        i, j = state_sequence[t], state_sequence[t + 1]
        ti, tj = transition_times[t], transition_times[t + 1]
        transition_count[i, j] += one(R2)
        duration[i] += tj - ti
    end
    duration[last(state_sequence)] += max_time(h) - last(transition_times)
    return ContinuousMarkovChainStats{R1,R2,R2}(
        initialization_count, transition_count, duration
    )
end
