function forward!(
    α::AbstractMatrix{R},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractControlledHMM,
    obs_density::AbstractMatrix{R},
    P_all::AbstractArray{<:Real,3},
) where {R<:Real}
    S, T = size(obs_density)
    p0 = initial_distribution(hmm)
    # Initialization
    α[:, 1] .= @views p0 .* obs_density[:, 1]
    α_sum_inv[1] = @views inv(sum(α[:, 1]))
    α[:, 1] .*= α_sum_inv[1]
    # Recursion
    @inbounds for t in 1:(T - 1)
        Pₜ = @view P_all[:, :, t]
        @inbounds for j in 1:S
            α[j, t + 1] = sum(α[i, t] * Pₜ[i, j] for i in 1:S) * obs_density[j, t + 1]
        end
        α_sum_inv[t + 1] = @views inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= α_sum_inv[t + 1]
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
    hmm::AbstractControlledHMM,
    obs_density::AbstractMatrix{R},
    P_all::AbstractArray{<:Real,3},
) where {R<:Real}
    S, T = size(obs_density)
    # Initialization
    β[:, T] .= one(R)
    # Recursion
    @inbounds for t in (T - 1):-1:1
        Pₜ = @view P_all[:, :, t]
        @inbounds for i in 1:S
            β[i, t] = sum(P[i, j] * obs_density[j, t + 1] * β[j, t + 1] for j in 1:S)
        end
        β[:, t] .*= α_sum_inv[t]
    end
    # Overflow check
    if @views any(all(iszero_safe, β[:, t]) for t in 1:T)
        throw(OverflowError("Log probabilities are too small in backward step."))
    end
    return nothing
end

function forward_backward!(
    α::AbstractMatrix{R},
    β::AbstractMatrix{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractControlledHMM,
    obs_density::AbstractMatrix{R},
    control_matrix::AbstractMatrix{<:Real},
    args...,
) where {R<:Real}
    S, T = size(obs_density)
    P_all = transition_matrix(hmm, control_matrix, args...)
    forward!(α, α_sum_inv, hmm, obs_density, P_all)
    backward!(β, α_sum_inv, hmm, obs_density, P_all)
    # State sufficient statistics
    @inbounds for t in 1:T
        γ[:, t] .= @views α[:, t] .* β[:, t]
        γ_sum_inv = @views inv(sum(γ[:, t]))
        γ[:, t] .*= γ_sum_inv
    end
    # Transitions sufficient statistics
    @inbounds for t in 1:(T - 1)
        Pₜ = @view P_all[:, :, t]
        @inbounds for j in 1:S
            @inbounds for i in 1:S
                ξ[i, j, t] = α[i, t] * Pₜ[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            end
        end
        ξ_sum_inv = @views inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξ_sum_inv
    end
    logL = -sum(log, α_sum_inv)
    return logL
end
