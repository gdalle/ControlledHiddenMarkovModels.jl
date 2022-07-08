function light_forward(
    hmm::AbstractHMM,
    obs_sequence::AbstractVector,
)
    S = nb_states(hmm)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm)
    P = transition_matrix(hmm)
    em = emissions(hmm)
    # Initialization
    α = [p0[i] * densityof(em[i], y₁) for i in 1:S]
    α_sum_inv = inv(sum(α))
    α .*= α_sum_inv
    logL = -log(α_sum_inv)
    # Recursion
    α_tmp = similar(α)
    @inbounds for t in 1:(T - 1)
        yₜ₊₁ = obs_sequence[t + 1]
        @inbounds for j in 1:S
            α_tmp[j] = sum(α[i] * P[i, j] for i in 1:S) * densityof(em[j], yₜ₊₁)
        end
        α .= α_tmp
        α_sum_inv = inv(sum(α))
        α .*= α_sum_inv
        logL -= log(α_sum_inv)
    end
    return α, logL
end

function forward!(
    α::AbstractMatrix{R},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    p0 = initial_distribution(hmm)
    P = transition_matrix(hmm)
    # Initialization
    α[:, 1] .= @views p0 .* obs_density[:, 1]
    α_sum_inv[1] = @views inv(sum(α[:, 1]))
    @views α[:, 1] .*= α_sum_inv[1]
    # Recursion
    @inbounds for t in 1:(T - 1)
        @inbounds for j in 1:S
            α[j, t + 1] = sum(α[i, t] * P[i, j] for i in 1:S) * obs_density[j, t + 1]
        end
        α_sum_inv[t + 1] = @views inv(sum(α[:, t + 1]))
        @views α[:, t + 1] .*= α_sum_inv[t + 1]
    end
    # Overflow check
    if @views any(all(iszero_safe, α[:, t]) for t in 1:T)
        throw(OverflowError("Probabilities are too small in forward step."))
    end
    return nothing
end

function backward!(
    β::AbstractMatrix{R},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)
    # Initialization
    β[:, T] .= one(R)
    # Recursion
    @inbounds for t in (T - 1):-1:1
        @inbounds for i in 1:S
            β[i, t] = sum(P[i, j] * obs_density[j, t + 1] * β[j, t + 1] for j in 1:S)
        end
        @views β[:, t] .*= α_sum_inv[t]
    end
    # Overflow check
    if @views any(all(iszero_safe, β[:, t]) for t in 1:T)
        throw(OverflowError("Log probabilities are too small in backward step."))
    end
    return nothing
end

"""
    forward_backward!(α, β, γ, ξ, α_sum_inv, γ_sum_inv, ξ_sum_inv, hmm, obs_density)

Apply the forward-backward algorithm in-place to update sufficient statistics.
"""
function forward_backward!(
    α::AbstractMatrix{R},
    β::AbstractMatrix{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)
    forward!(α, α_sum_inv, hmm, obs_density)
    backward!(β, α_sum_inv, hmm, obs_density)
    # State sufficient statistics
    @inbounds for t in 1:T
        @views γ[:, t] .= α[:, t] .* β[:, t]
        γ_sum_inv = @views inv(sum(γ[:, t]))
        @views γ[:, t] .*= γ_sum_inv
    end
    # Transitions sufficient statistics
    @inbounds for t in 1:(T - 1)
        @inbounds for j in 1:S
            @inbounds for i in 1:S
                ξ[i, j, t] = α[i, t] * P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            end
        end
        ξ_sum_inv = @views inv(sum(ξ[:, :, t]))
        @views ξ[:, :, t] .*= ξ_sum_inv
    end
    logL = -sum(log, α_sum_inv)
    return Float64(logL)
end
