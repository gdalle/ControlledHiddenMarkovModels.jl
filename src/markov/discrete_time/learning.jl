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
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
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
        initialization_count .+= @view γ[k][:, 1]
        transition_count .+= dropdims(sum(ξ[k], dims=3), dims=3)
    end
    return DiscreteMarkovChainStats{R1,R2}(; initialization_count, transition_count)
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
    mctype::Type{DiscreteMarkovChain{R1,R2}}, x::AbstractVector{<:Integer}; kwargs...
) where {R1,R2}
    ss = suffstats(mctype, x; kwargs...)
    return fit_mle(mctype, ss)
end

function Distributions.fit_mle(
    mctype::Type{DiscreteMarkovChain{R1,R2}},
    γ::Vector{<:AbstractMatrix{<:Real}},
    ξ::Vector{<:AbstractArray{<:Real,3}};
    kwargs...,
) where {R1,R2}
    ss = suffstats(mctype, γ, ξ; kwargs...)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{DiscreteMarkovChain{R1,R2}},
    prior::DiscreteMarkovChainPrior,
    x::AbstractVector{<:Integer};
    kwargs...,
) where {R1,R2}
    ss = suffstats(mctype, x; kwargs...)
    ss.initialization_count .+= (prior.π0_α .- 1)
    ss.transition_count .+= (prior.P_α .- 1)
    return fit_mle(mctype, ss)
end

function fit_map(
    mctype::Type{DiscreteMarkovChain{R1,R2}},
    prior::DiscreteMarkovChainPrior,
    γ::Vector{<:AbstractMatrix{<:Real}},
    ξ::Vector{<:AbstractArray{<:Real,3}};
    kwargs...,
) where {R1,R2}
    ss = suffstats(mctype, γ, ξ; kwargs...)
    ss.initialization_count .+= (prior.π0_α .- 1)
    ss.transition_count .+= (prior.P_α .- 1)
    return fit_mle(mctype, ss)
end
