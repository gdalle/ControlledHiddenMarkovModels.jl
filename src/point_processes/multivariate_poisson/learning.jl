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

## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{MultivariatePoissonProcess{R}}, ss::MultivariatePoissonProcessStats
) where {R<:Real}
    λ = convert(Vector{R}, ss.event_count ./ ss.duration)
    return MultivariatePoissonProcess(λ)
end

function fit_map(
    pptype::Type{MultivariatePoissonProcess{R}},
    prior::MultivariatePoissonProcessPrior,
    ss::MultivariatePoissonProcessStats,
) where {R}
    (; λ_α, λ_β) = prior
    posterior_event_count = ss.event_count .+ λ_α .- one(eltype(λ_α))
    posterior_duration = ss.duration + λ_β
    ss_posterior = MultivariatePoissonProcessStats(
        posterior_event_count, posterior_duration
    )
    return fit_mle(pptype, ss_posterior)
end

## Fit from observations

function Distributions.fit_mle(
    pptype::Type{MultivariatePoissonProcess{R}}, args...; kwargs...
) where {R}
    ss = suffstats(pptype, args...; kwargs...)
    return fit_mle(pptype, ss)
end

function fit_map(
    pptype::Type{MultivariatePoissonProcess{R}},
    prior::MultivariatePoissonProcessPrior,
    args...;
    kwargs...,
) where {R}
    ss = suffstats(pptype, args..., kwargs...)
    return fit_map(pptype, prior, ss)
end
