## Types

suffstats_type(::Type{D}) where {D<:Normal} = Distributions.NormalStats

function suffstats_type(::Type{M}) where {R1<:Real,R2<:Real,M<:DiscreteMarkovChain{R1,R2}}
    return DiscreteMarkovChainStats{R1,R2}
end

function suffstats_type(::Type{P}) where {R<:Real,P<:MultivariatePoissonProcess{R}}
    return MultivariatePoissonProcessStats{R,R}
end

## Addition

function add_suffstats(ss1::Distributions.NormalStats, ss2::Distributions.NormalStats)
    s = ss1.s + ss2.s
    m = (ss1.tw * ss1.m + ss2.tw * ss2.m) / (ss1.tw + ss2.tw)
    s2 = ss1.s2 + ss2.s2
    tw = ss1.tw + ss2.tw
    return Distributions.NormalStats(s, m, s2, tw)
end

function add_suffstats(
    ss1::DiscreteMarkovChainStats{R1,R2}, ss2::DiscreteMarkovChainStats{R1,R2}
) where {R1<:Real,R2<:Real}
    initialization_count = ss1.initialization_count .+ ss2.initialization_count
    transition_count = ss1.transition_count .+ ss2.transition_count
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

function add_suffstats(
    ss1::MultivariatePoissonProcessStats{R1,R2}, ss2::MultivariatePoissonProcessStats{R1,R2}
) where {R1<:Real,R2<:Real}
    event_count = ss1.event_count .+ ss2.event_count
    duration = ss1.duration + ss2.duration
    return MultivariatePoissonProcessStats{R1,R2}(event_count, duration)
end
