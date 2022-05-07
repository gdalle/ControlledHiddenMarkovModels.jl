## Sufficient statistics

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}}, x::AbstractVector{<:Integer}
) where {R1,R2}
    S, T = maximum(x), length(x)
    initialization_count = zeros(R1, S)
    initialization_count[x[1]] = one(R1)
    transition_count = zeros(R2, S, S)
    for t in 1:(T - 1)
        transition_count[x[t], x[t + 1]] += one(R2)
    end
    return DiscreteMarkovChainStats(;
        initialization_count=initialization_count, transition_count=transition_count
    )
end

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}},
    γ::Vector{<:AbstractMatrix{<:Real}},
    ξ::Vector{<:AbstractArray{<:Real,3}},
) where {R1,R2}
    K = length(γ)
    S = size(γ[1], 1)
    initialization_count = zeros(R1, S)
    transition_count = zeros(R2, S, S)
    for k in 1:K
        γₖ = γ[k]
        ξₖ = ξ[k]
        Tₖ = size(γₖ, 2)
        for i in 1:S
            initialization_count[i] += γₖ[i, 1]
        end
        for i in 1:S, j in 1:S, t in 1:(Tₖ - 1)
            transition_count[i, j] += ξₖ[i, j, t]
        end
    end
    return DiscreteMarkovChainStats(;
        initialization_count=initialization_count, transition_count=transition_count
    )
end

## Fit

function Distributions.fit_mle(
    ::Type{DiscreteMarkovChain{R1,R2}}, ss::DiscreteMarkovChainStats
) where {R1,R2}
    π0 = ss.initialization_count ./ sum(ss.initialization_count)
    P = ss.transition_count ./ sum(ss.transition_count; dims=2)
    return DiscreteMarkovChain{R1,R2}(π0, P)
end

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
    ss.initialization .+= (prior.π0α .- 1)
    ss.transition_count .+= (prior.Pα .- 1)
    return fit_mle(mctype, ss)
end
