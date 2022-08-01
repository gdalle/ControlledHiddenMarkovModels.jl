## Forward-backward

function forward_turbo!(
    α::AbstractMatrix{R},
    α_sum::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    p0, P = initial_distribution(hmm), transition_matrix(hmm)
    @turbo α .= zero(R)

    # Initialization
    @turbo view(α, :, 1) .= p0 .* view(obs_density, :, 1)
    α_sum[1] = sum(view(α, :, 1))
    @turbo view(α, :, 1) ./= α_sum[1]

    # Recursion
    for t in 1:(T - 1)
        @turbo for j in 1:S, i in 1:S
            α[j, t + 1] += α[i, t] * P[i, j] * obs_density[j, t + 1]
        end
        α_sum[t + 1] = sum(view(α, :, t + 1))
        @turbo view(α, :, t + 1) ./= α_sum[t + 1]
    end

    # Overflow check
    for t in 1:T
        if all(iszero_safe, view(α, :, t))
            throw(OverflowError("Probabilities are too small in forward step."))
        end
    end

    return nothing
end

function backward_turbo!(
    β::AbstractMatrix{R},
    α_sum::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)
    @turbo β .= zero(R)

    # Initialization
    @turbo view(β, :, T) .= one(R)

    # Recursion
    for t in (T - 1):-1:1
        @turbo for i in 1:S, j in 1:S
            β[i, t] += P[i, j] * obs_density[j, t + 1] * β[j, t + 1] / α_sum[t]
        end
    end

    # Overflow check
    for t in 1:T
        if all(iszero_safe, view(β, :, t))
            throw(OverflowError("Log probabilities are too small in backward step."))
        end
    end

    return nothing
end

function forward_backward_turbo!(
    α::AbstractMatrix{R},
    β::AbstractMatrix{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    α_sum::AbstractVector{R},
    γ_sum::AbstractVector{R},
    ξ_sum::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    forward!(α, α_sum, hmm, obs_density)
    backward!(β, α_sum, hmm, obs_density)

    # State sufficient statistics
    @turbo γ_sum .= zero(R)

    @turbo for t in 1:T, i in 1:S
        γ[i, t] = α[i, t] * β[i, t]
        γ_sum[t] += γ[i, t]
    end

    @turbo for t in 1:T, i in 1:S
        γ[i, t] /= γ_sum[t]
    end

    # Transitions sufficient statistics
    @turbo ξ_sum .= zero(R)

    @turbo for t in 1:(T - 1), j in 1:S, i in 1:S
        ξ[i, j, t] = α[i, t] * P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
        ξ_sum[t] += ξ[i, j, t]
    end

    @turbo for t in 1:(T - 1), j in 1:S, i in 1:S
        ξ[i, j, t] /= ξ_sum[t]
    end

    logL = zero(float(R))
    @turbo for t in 1:T
        logL += log(α_sum[t])
    end

    return logL
end
