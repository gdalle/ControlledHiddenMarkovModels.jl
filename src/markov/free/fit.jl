## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{MarkovChain{R1,R2}}, ss::MarkovChainStats
) where {R1,R2}
    p0 = ss.initialization_count ./ sum(ss.initialization_count)
    P = ss.transition_count ./ sum(ss.transition_count; dims=2)
    return MarkovChain{R1,R2}(p0, P)
end

function fit_map(
    mctype::Type{MarkovChain{R1,R2}},
    prior::MarkovChainPrior,
    ss::MarkovChainStats;
    kwargs...,
) where {R1,R2}
    (; p0_α, P_α) = prior
    initialization_count = ss.initialization_count .+ p0_α .- one(eltype(p0_α))
    transition_count = ss.transition_count .+ P_α .- one(eltype(P_α))
    ss_posterior = MarkovChainStats(initialization_count, transition_count)
    return fit_mle(mctype, ss_posterior)
end

## Fit from observations

function Distributions.fit_mle(
    mctype::Type{MarkovChain{R1,R2}}, args...; kwargs...
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{MarkovChain{R1,R2}}, prior::MarkovChainPrior, args...; kwargs...
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    return fit_map(mctype, prior, ss)
end
