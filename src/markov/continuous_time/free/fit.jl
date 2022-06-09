## Fit from sufficient stats

function Distributions.fit_mle(
    ::Type{ContinuousMarkovChain{R1,R2}}, ss::ContinuousMarkovChainStats
) where {R1,R2}
    p0 = ss.initialization_count ./ sum(ss.initialization_count)
    Q = ss.transition_count ./ ss.duration
    Q[diagind(Q)] .= zero(R2)
    Q[diagind(Q)] .= -dropdims(sum(Q; dims=2); dims=2)
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
