## Likelihood of obs_sequence

"""
    update_obs_density!(obs_density, hmm, obs_sequence)

Set `obs_density[s, t]` to the likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix{R}, hmm::HMM, obs_sequence::AbstractVector
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(emission(hmm, s), oₜ)
        end
    end
    for t in 1:T
        if all_zero(@view obs_density[:, t])
            throw(OverflowError("Densities are too small for observations."))
        end
    end
end

function compute_obs_density(
    hmm::HMM{Tr,Em}, obs_sequence::AbstractVector{O},
) where {Tr,Em,O}
    T, S = length(obs_sequence), nb_states(hmm)
    obs_density = [densityof(emission(hmm, s), obs_sequence[t]) for s = 1:S, t = 1:T]
    for t in 1:T
        if all_zero(@view obs_density[:, t])
            throw(OverflowError("Densities are too small for observations."))
        end
    end
    return obs_density
end

## Forward-backward

function forward!(
    α::AbstractMatrix{R}, c::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    π0, P = initial_distribution(hmm), transition_matrix(hmm)
    for i in 1:S
        α[i, 1] = π0[i] * obs_density[i, 1]
    end
    c[1] = sum(@view α[:, 1])  # scaling
    for i in 1:S
        α[i, 1] /= c[1]
    end
    for t in 1:(T - 1)
        for j in 1:S
            α[j, t + 1] = sum(α[i, t] * P[i, j] for i in 1:S) * obs_density[j, t + 1]
        end
        c[t + 1] = sum(@view α[:, t + 1])  # scaling
        for j in 1:S
            α[j, t + 1] /= c[t + 1]
        end
    end
    for t in 1:T
        if all_zero(@view α[:, t])
            throw(OverflowError("Probabilities are too small in forward step."))
        end
    end
    return nothing
end

function backward!(
    β::AbstractMatrix{R}, c::AbstractVector{R}, hmm::HMM, obs_density::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    for i in 1:S
        β[i, T] = one(R)
    end
    for t in (T - 1):-1:1
        for i in 1:S
            β[i, t] = sum(P[i, j] * obs_density[j, t + 1] * β[j, t + 1] for j in 1:S) / c[t]
        end
    end
    for t in 1:T
        if all_zero(@view β[:, t])
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
    c::AbstractVector{R},
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    hmm::HMM,
    obs_density::AbstractMatrix{R},
) where {R<:Real}
    S, T = size(obs_density)
    P = transition_matrix(hmm)

    forward!(α, c, hmm, obs_density)
    backward!(β, c, hmm, obs_density)

    for t in 1:T
        for i in 1:S
            γ[i, t] = α[i, t] * β[i, t]
        end
        sumγ = sum(@view γ[:, t])
        for i in 1:S
            γ[i, t] /= sumγ
        end
    end

    for t in 1:(T - 1)
        for j in 1:S, i in 1:S
            ξ[i, j, t] = α[i, t] * P[i, j] * obs_density[j, t + 1] * β[j, t + 1]
        end
        sumξ = sum(@view ξ[:, :, t])
        for j in 1:S, i in 1:S
            ξ[i, j, t] /= sumξ
        end
    end

    logL = sum(log, c)

    return logL
end
