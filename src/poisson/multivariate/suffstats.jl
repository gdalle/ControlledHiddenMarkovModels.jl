struct MultivariatePoissonProcessStats{R1<:Real,R2<:Real}
    event_count::Vector{R1}
    duration::R2
end

## Compute sufficient stats

function Distributions.suffstats(
    ::Type{MultivariatePoissonProcess{R}}, history::History{<:Integer,<:Real}
) where {R<:Real}
    M = max_mark(history)
    event_count = zeros(Int, M)
    for m in event_marks(history)
        event_count[m] += 1
    end
    return MultivariatePoissonProcessStats(event_count, duration(history))
end

function Distributions.suffstats(
    ::Type{MultivariatePoissonProcess{R}},
    histories::AbstractVector{<:History{<:Integer,<:Real}},
    weights::AbstractVector{W},
) where {R<:Real,W<:Real}
    M = mapreduce(max_mark, max, histories)
    total_event_count = zeros(W, M)
    total_duration = zero(W)
    for (history, weight) in zip(histories, weights)
        total_duration += weight * duration(history)
        for m in event_marks(history)
            total_event_count[m] += weight
        end
    end
    return MultivariatePoissonProcessStats(total_event_count, total_duration)
end
