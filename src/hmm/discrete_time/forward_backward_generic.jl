## Forward-backward

function forward!(
    α::AbstractMatrix{R}, α_sum::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    p0, P = initial_distribution(hmm), transition_matrix(hmm)

    # Initialization
    α_sum[1] = zero(R)
    for i in 1:S
        α[i, 1] = p0[i] * obs_density[i, 1]
        α_sum[1] += α[i, 1]
    end
    for i in 1:S
        α[i, 1] /= α_sum[1]
    end

    # Recursion
    for t in 1:(T - 1)
        α_sum[t + 1] = zero(R)
        for j in 1:S
            α[j, t + 1] = zero(R)
            for i in 1:S
                α[j, t + 1] += α[i, t] * P[i, j]
            end
            α[j, t + 1] *= obs_density[j, t + 1]
            α_sum[t + 1] += α[j, t + 1]
        end
        for j in 1:S
            α[j, t + 1] /= α_sum[t + 1]
        end
    end

    # Overflow check
    for t in 1:T
        if all(iszero_safe, view(α, :, t))
            throw(OverflowError("Probabilities are too small in forward step."))
        end
    end

    return nothing
end

function backward!(
    β::AbstractMatrix{R}, α_sum::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    # Initialization
    for i in 1:S
        β[i, T] = one(R)
    end

    # Recursion
    for t in (T - 1):-1:1
        for i in 1:S
            β[i, t] = zero(R)
            for j in 1:S
                β[i, t] += P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            end
            β[i, t] /= α_sum[t]
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

"""
    forward_backward!(α, β, γ, ξ, α_sum, γ_sum, ξ_sum, hmm, obs_density)

Apply the forward-backward algorithm in-place to update sufficient statistics.
"""
function forward_backward!(
    α::AbstractMatrix{R},
    β::AbstractMatrix{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    α_sum::AbstractVector{R},
    γ_sum::AbstractVector{R},
    ξ_sum::AbstractVector{R},
    hmm::HMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    forward!(α, α_sum, hmm, obs_density)
    backward!(β, α_sum, hmm, obs_density)

    # State sufficient statistics
    for t in 1:T
        γ_sum[t] = zero(R)
        for i in 1:S
            γ[i, t] = α[i, t] * β[i, t]
            γ_sum[t] += γ[i, t]
        end
        for i in 1:S
            γ[i, t] /= γ_sum[t]
        end
    end

    # Transitions sufficient statistics
    for t in 1:(T - 1)
        ξ_sum[t] = zero(R)
        for j in 1:S, i in 1:S
            ξ[i, j, t] = α[i, t] * P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            ξ_sum[t] += ξ[i, j, t]
        end
        for j in 1:S, i in 1:S
            ξ[i, j, t] /= ξ_sum[t]
        end
    end

    logL = zero(float(R))
    for t in 1:T
        logL += log(α_sum[t])
    end

    return logL
end
