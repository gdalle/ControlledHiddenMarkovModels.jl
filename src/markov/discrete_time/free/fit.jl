## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{DiscreteMarkovChain{R1,R2}}, ss::DiscreteMarkovChainStats
) where {R1,R2}
    p0 = ss.initialization_count ./ sum(ss.initialization_count)
    P = ss.transition_count ./ sum(ss.transition_count; dims=2)
    return DiscreteMarkovChain{R1,R2}(p0, P)
end

function fit_map(
    mctype::Type{DiscreteMarkovChain{R1,R2}},
    prior::DiscreteMarkovChainPrior,
    ss::DiscreteMarkovChainStats;
    kwargs...,
) where {R1,R2}
    (; p0_α, P_α) = prior
    initialization_count = ss.initialization_count .+ p0_α .- one(eltype(p0_α))
    transition_count = ss.transition_count .+ P_α .- one(eltype(P_α))
    ss_posterior = DiscreteMarkovChainStats(initialization_count, transition_count)
    return fit_mle(mctype, ss_posterior)
end

## Fit from observations

function Distributions.fit_mle(
    mctype::Type{DiscreteMarkovChain{R1,R2}}, args...; kwargs...
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{DiscreteMarkovChain{R1,R2}},
    prior::DiscreteMarkovChainPrior,
    args...;
    kwargs...,
) where {R1,R2}
    ss = suffstats(mctype, args...; kwargs...)
    return fit_map(mctype, prior, ss)
end
