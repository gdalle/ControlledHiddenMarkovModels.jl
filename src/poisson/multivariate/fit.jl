## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{MultivariatePoissonProcess{R,V}}, ss::MultivariatePoissonProcessStats
) where {R<:Real,V}
    λ = convert(V, ss.event_count ./ ss.duration)
    return MultivariatePoissonProcess(λ)
end

function fit_map(
    pptype::Type{MultivariatePoissonProcess{R,V}},
    prior::MultivariatePoissonProcessPrior,
    ss::MultivariatePoissonProcessStats,
) where {R,V}
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
    pptype::Type{MultivariatePoissonProcess{R,V}}, args...; kwargs...
) where {R,V}
    ss = suffstats(pptype, args...; kwargs...)
    return fit_mle(pptype, ss)
end

function fit_map(
    pptype::Type{MultivariatePoissonProcess{R,V}},
    prior::MultivariatePoissonProcessPrior,
    args...;
    kwargs...,
) where {R,V}
    ss = suffstats(pptype, args..., kwargs...)
    return fit_map(pptype, prior, ss)
end
