## Compute sufficient stats

function Distributions.suffstats(
    ::Type{MultivariatePoissonProcess{R}}, h::History
) where {R<:Real}
    event_count = zeros(Int, maximum(event_marks(h)))
    for m in event_marks(h)
        event_count[m] += 1
    end
    return MultivariatePoissonProcessStats(; duration=duration(h), event_count=event_count)
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
    ss_posterior = MultivariatePoissonProcessStats(;
        event_count=ss.event_count .+ λ_α .- one(eltype(λ_α)), duration=ss.duration + λ_β
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
