## Compute sufficient stats

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}}, state_sequence::AbstractVector{<:Integer}
) where {R1,R2}
    S, T = maximum(state_sequence), length(state_sequence)
    initialization_count = zeros(R1, S)
    initialization_count[state_sequence[1]] = one(R1)
    transition_count = zeros(R2, S, S)
    for t in 1:(T - 1)
        transition_count[state_sequence[t], state_sequence[t + 1]] += one(R2)
    end
    return DiscreteMarkovChainStats{R1,R2}(initialization_count, transition_count)
end

function Distributions.suffstats(
    ::Type{DiscreteMarkovChain{R1,R2}},
    state_sequences::AbstractVector{<:AbstractVector{<:Integer}},
) where {R1,R2}
    S = mapreduce(maximum, max, state_sequences)
    initialization_count = zeros(R1, S)
    transition_count = zeros(R2, S, S)
    for state_sequence in state_sequences
        initialization_count[state_sequence[1]] += one(R1)
        T = length(state_sequence)
        for t in 1:(T - 1)
            transition_count[state_sequence[t], state_sequence[t + 1]] += one(R2)
        end
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
        transition_count .+= dropdims(sum(ξ[k]; dims=3); dims=3)
    end
    return DiscreteMarkovChainStats{R1,R2}(; initialization_count, transition_count)
end

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
    ss_posterior = DiscreteMarkovChainStats(;
        initialization_count=ss.initialization_count .+ p0_α .- one(eltype(p0_α)),
        transition_count=ss.transition_count .+ P_α .- one(eltype(P_α)),
    )
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
