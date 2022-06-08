## Compute sufficient stats

function Distributions.suffstats(
    ::Type{ContinuousMarkovChain{R1,R2}},
    initialization_count::AbstractVector{<:Real},
    transition_count::AbstractMatrix{<:Real},
    duration::AbstractVector{<:Real},
) where {R1,R2}
    return ContinuousMarkovChainStats{R1,R2}(
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

## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{ContinuousMarkovChain{R1,R2}}, ss::ContinuousMarkovChainStats
) where {R1,R2}
    p0 = ss.initialization_count ./ sum(ss.initialization_count)
    Q = ss.transition_count ./ ss.duration
    Q[diagind(Q)] .-= dropdims(sum(Q; dims=2); dims=2)
    return ContinuousMarkovChain{R1,R2}(p0, Q)
end

## Fit from observations

function Distributions.fit_mle(
    mctype::Type{ContinuousMarkovChain{R1,R2}}, args...; kwargs...
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{ContinuousMarkovChain{R1,R2}},
    prior::ContinuousMarkovChainPrior,
    args...;
    kwargs...,
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    ss.initialization .+= (prior.p0α .- 1)
    ss.transition_count .+= (prior.Pα .- 1)
    ss.duration .+= prior.Pβ
    return fit_mle(mctype, ss)
end
