## Likelihood of obs_sequence

"""
    update_obs_density!(obs_density, hmm, obs_sequence)

Set `obs_density[s, t]` to the likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix{R}, hmm::HMM, obs_sequence::AbstractVector
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    @turbo for t in 1:T, s in 1:S
        obs_density[s, t] = densityof(emission(hmm, s), obs_sequence[t])
    end
    @threads for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
end

function compute_obs_density(
    hmm::HMM{Tr,Em}, obs_sequence::AbstractVector{O}
) where {Tr,Em,O}
    T, S = length(obs_sequence), nb_states(hmm)
    obs_density = [densityof(emission(hmm, s), obs_sequence[t]) for s in 1:S, t in 1:T]
    @threads for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
    return obs_density
end

## Forward-backward

function forward!(
    α::AbstractMatrix{R}, sumα::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_density)
    p0, P = initial_distribution(hmm), transition_matrix(hmm)
    @turbo α .= zero(R)
    @turbo sumα .= zero(R)

    # Initialization
    @turbo view(α, :, 1) .= p0 .* view(obs_density, :, 1)
    sumα[1] = sum(view(α, :, 1))
    @turbo view(α, :, 1) ./= sumα[1]

    # Recursion
    for t in 1:(T - 1)
        @turbo for j in 1:S, i in 1:S
            α[j, t + 1] += α[i, t] * P[i, j] * obs_density[j, t + 1]
        end
        sumα[t + 1] = sum(view(α, :, t + 1))
        @turbo view(α, :, t + 1) ./= sumα[t + 1]
    end
    for t in 1:T
        if all(iszero_safe, view(α, :, t))
            throw(OverflowError("Probabilities are too small in forward step."))
        end
    end
    return nothing
end

function backward!(
    β::AbstractMatrix{R}, sumα::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)
    @turbo β .= zero(R)

    @turbo view(β, :, T) .= one(R)
    for t in (T - 1):-1:1
        @turbo for i in 1:S, j in 1:S
            β[i, t] += P[i, j] * obs_density[j, t + 1] * β[j, t + 1] / sumα[t]
        end
    end
    for t in 1:T
        if all(iszero_safe, view(β, :, t))
            throw(OverflowError("Log probabilities are too small in backward step."))
        end
    end
    return nothing
end

"""
    forward_backward!(α, β, c, γ, ξ, hmm, obs_density)

Apply the forward-backward algorithm in-place to update sufficient statistics.
"""
function forward_backward!(
    α::AbstractMatrix{R},
    β::AbstractMatrix{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    sumα::AbstractVector{R},
    sumγ::AbstractVector{R},
    sumξ::AbstractVector{R},
    hmm::HMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    forward!(α, sumα, hmm, obs_density)
    backward!(β, sumα, hmm, obs_density)

    @turbo sumγ .= zero(R)
    @turbo sumξ .= zero(R)

    @turbo for t in 1:T, i in 1:S
        γ[i, t] = α[i, t] * β[i, t]
        sumγ[t] += γ[i, t]
    end

    @turbo for t in 1:T, i in 1:S
        γ[i, t] /= sumγ[t]
    end

    @turbo for t in 1:(T - 1), j in 1:S, i in 1:S
        ξ[i, j, t] = α[i, t] * P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
        sumξ[t] += ξ[i, j, t]
    end

    @turbo for t in 1:(T - 1), j in 1:S, i in 1:S
        ξ[i, j, t] /= sumξ[t]
    end

    logL = zero(float(R))
    @turbo for t in 1:T
        logL += log(sumα[t])
    end

    return logL
end
