## Sufficient statistics

function Distributions.suffstats(
    ::Type{<:DiscreteMarkovChain}, x::AbstractVector{<:Integer}
)
    S, T = maximum(x), length(x)
    initialization_count = [s == x[1] for s in 1:S]
    transition_count = zeros(Int, S, S)
    for t in 1:(T - 1)
        transition_count[x[t], x[t + 1]] += 1
    end
    return DiscreteMarkovChainStats(;
        initialization_count=initialization_count, transition_count=transition_count
    )
end

function Distributions.suffstats(
    ::Type{<:DiscreteMarkovChain}, γ::AbstractMatrix{<:Real}, ξ::AbstractArray{<:Real,3}
)
    T, S = size(γ)
    initialization_count = γ[1, :]
    transition_count = zeros(Float64, S, S)
    for i in 1:S, j in 1:S, t in 1:(T - 1)
        transition_count[i, j] += ξ[t, i, j]
    end
    return DiscreteMarkovChainStats(;
        initialization_count=initialization_count, transition_count=transition_count
    )
end

function Distributions.suffstats(
    ::Type{<:DiscreteMarkovChain},
    γ::Vector{<:AbstractMatrix{<:Real}},
    ξ::Vector{<:AbstractArray{<:Real,3}},
)
    K = length(γ)
    T = [size(γ[k], 1) for k in 1:K]
    S = size(γ[1], 2)

    initialization_count = sum(γ[k][1, :] for k in 1:K)
    transition_count = zeros(Float64, S, S)
    for i in 1:S, j in 1:S, k in 1:K, t in 1:(T[k] - 1)
        transition_count[i, j] += ξ[k][t, i, j]
    end
    return DiscreteMarkovChainStats(;
        initialization_count=initialization_count, transition_count=transition_count
    )
end

## Fit

function Distributions.fit_mle(::Type{<:DiscreteMarkovChain}, ss::DiscreteMarkovChainStats)
    π0 = ss.initialization_count ./ sum(ss.initialization_count)
    P = ss.transition_count ./ sum(ss.transition_count; dims=2)
    return DiscreteMarkovChain(; π0=π0, P=P)
end

function Distributions.fit_mle(mctype::Type{<:DiscreteMarkovChain}, args...; kwargs...)
    ss = suffstats(mctype, args...; kwargs...)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{<:DiscreteMarkovChain}, prior::DiscreteMarkovChainPrior, args...; kwargs...
)
    ss = suffstats(mctype, args...; kwargs...)
    ss.initialization .+= (prior.π0α .- 1)
    ss.transition_count .+= (prior.Pα .- 1)
    return fit_mle(mctype, ss)
end
