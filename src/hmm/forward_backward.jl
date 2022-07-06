## Forward-backward

function forward!(
    α::AbstractMatrix{R},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
    control_sequence::AbstractVector=Fill(nothing, size(obs_density, 2)),
    ps=nothing,
    st=nothing,
) where {R<:Real}
    S, T = size(obs_density)
    p0 = initial_distribution(hmm)

    # Initialization
    for i in 1:S
        α[i, 1] = p0[i] * obs_density[i, 1]
    end
    α_sum_inv[1] = inv(sum(view(α, :, 1)))
    for i in 1:S
        α[i, 1] *= α_sum_inv[1]
    end

    # Recursion
    @inbounds for t in 1:(T - 1)
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, ps, st)
        @inbounds for j in 1:S
            tmp = zero(R)
            @inbounds for i in 1:S
                tmp += α[i, t] * Pₜ[i, j]
            end
            α[j, t + 1] = tmp * obs_density[j, t + 1]
        end
        α_sum_inv[t + 1] = inv(sum(view(α, :, t + 1)))
        @inbounds for j in 1:S
            α[j, t + 1] *= α_sum_inv[t + 1]
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
    β::AbstractMatrix{R},
    α_sum_inv::AbstractVector{R},
    hmm::AbstractHMM,
    obs_density::AbstractMatrix{R},
    control_sequence::AbstractVector=Fill(nothing, size(obs_density, 2)),
    ps=nothing,
    st=nothing,
) where {R<:Real}
    S, T = size(obs_density)

    # Initialization
    for i in 1:S
        β[i, T] = one(R)
    end

    # Recursion
    @inbounds for t in (T - 1):-1:1
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, ps, st)
        @inbounds for i in 1:S
            tmp = zero(R)
            @inbounds for j in 1:S
                tmp += Pₜ[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            end
            β[i, t] = tmp * α_sum_inv[t]
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
    forward_backward!(
        α, β, γ, ξ, α_sum_inv, γ_sum_inv, ξ_sum_inv, hmm, obs_density,
        [control_sequence, ps, st]
    )

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
    control_sequence::AbstractVector=Fill(nothing, size(obs_density, 2)),
    ps=nothing,
    st=nothing,
) where {R<:Real}
    S, T = size(obs_density)

    forward!(α, α_sum_inv, hmm, obs_density, control_sequence, ps, st)
    backward!(β, α_sum_inv, hmm, obs_density, control_sequence, ps, st)

    # State sufficient statistics
    @inbounds for t in 1:T
        @inbounds for i in 1:S
            γ[i, t] = α[i, t] * β[i, t]
        end
        γ_sum_inv = inv(sum(view(γ, :, t)))
        @inbounds for i in 1:S
            γ[i, t] *= γ_sum_inv
        end
    end

    # Transitions sufficient statistics
    @inbounds for t in 1:(T - 1)
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, ps, st)
        @inbounds for j in 1:S
            @inbounds for i in 1:S
                ξ[i, j, t] = α[i, t] * Pₜ[i, j] * obs_density[j, t + 1] * β[j, t + 1]
            end
        end
        ξ_sum_inv = inv(sum(view(ξ, :, :, t)))
        @inbounds for j in 1:S
            @inbounds for i in 1:S
                ξ[i, j, t] *= ξ_sum_inv
            end
        end
    end

    logL = zero(float(R))
    @inbounds for t in 1:T
        logL -= log(α_sum_inv[t])
    end

    return logL
end

struct ForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    β::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
    α_sum_inv::Vector{Vector{R}}
end

function initialize_forward_backward_multiple_sequences(
    obs_densities::AbstractVector{<:AbstractMatrix{R}}
) where {R<:Real}
    K = length(obs_densities)
    S = size(obs_densities[1], 1)
    T = [size(obs_densities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    α_sum_inv = [Vector{R}(undef, T[k]) for k in 1:K]
    fb_storage = ForwardBackwardStorage{R}(α, β, γ, ξ, α_sum_inv)
    return fb_storage
end
