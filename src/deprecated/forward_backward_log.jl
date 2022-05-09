## Likelihood of obs_sequence

"""
    update_obs_logdensity!(obs_density, hmm, obs_sequence)

Set `obs_logdensity[s, t]` to the log-likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_logdensity!(
    obs_logdensity::AbstractMatrix{R}, hmm::HMM, obs_sequence::AbstractVector
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_logdensity[s, t] = logdensityof(emission(hmm, s), oₜ)
        end
    end
    for t in 1:T
        if all_minus_inf(@view obs_logdensity[:, t])
            throw(OverflowError("Log-densities are too small for obs_sequence."))
        end
    end
end

function compute_obs_logdensity(
    hmm::HMM{Tr,Em}, obs_sequence::AbstractVector{O}
) where {Tr,Em,O}
    T, S = length(obs_sequence), nb_states(hmm)
    obs_logdensity = [
        logdensityof(emission(hmm, s), obs_sequence[t]) for s in 1:S, t in 1:T
    ]
    for t in 1:T
        if all_minus_inf(@view obs_logdensity[:, t])
            throw(OverflowError("Densities are too small for observations."))
        end
    end
    return obs_density
end

## Forward-backward

function forward_log!(
    logα::AbstractMatrix{R}, hmm::HMM, obs_logdensity::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_logdensity)
    logp0, logP = log.(initial_distribution(hmm)), log.(transition_matrix(hmm))
    for i in 1:S
        logα[i, 1] = logp0[i] + obs_logdensity[i, 1]
    end
    for t in 1:(T - 1)
        for j in 1:S
            # TODO: no allocations
            logα[j, t + 1] = (
                logsumexp(view(logα, :, t) + view(logP, :, j)) + obs_logdensity[j, t + 1]
            )
        end
    end
    for t in 1:T
        if all(isnan, @view logα[t, :])
            throw(OverflowError("Log probabilities are too small in forward step."))
        end
    end
    return nothing
end

function backward_log!(
    logβ::AbstractMatrix{R}, hmm::HMM, obs_logdensity::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_logdensity)
    logP = log.(transition_matrix(hmm))

    for i in 1:S
        logβ[i, T] = zero(R)
    end
    for t in (T - 1):-1:1
        for i in 1:S
            # TODO: no allocations
            logβ[t, i] = logsumexp(
                view(logP, i, :) .+ view(obs_logdensity, :, t + 1) .+ view(logβ, :, t + 1)
            )
        end
    end
    for t in 1:T
        if all(isnan, @view logβ[t, :])
            throw(OverflowError("Log probabilities are too small in backward step."))
        end
    end
    return nothing
end

"""
    forward_backward_log!(logα, logβ, logγ, logξ, hmm, obs_logdensity)

Apply the logarithmic forward-backward algorithm in-place to update sufficient statistics.
"""
function forward_backward_log!(
    logα::AbstractMatrix{R},
    logβ::AbstractMatrix{R},
    logγ::AbstractMatrix{R},
    logξ::AbstractArray{R,3},
    hmm::HMM,
    obs_logdensity::AbstractMatrix,
) where {R<:Real}
    S, T = size(obs_logdensity)
    logP = log.(transition_matrix(hmm))

    forward_log!(logα, hmm, obs_logdensity)
    backward_log!(logβ, hmm, obs_logdensity)

    for t in 1:T
        for i in 1:S
            logγ[i, t] = logα[i, t] + logβ[i, t]
        end
        logsumγ = logsumexp(@view logγ[:, t])
        for i in 1:S
            logγ[i, t] -= logsumγ
        end
    end

    for t in 1:(T - 1)
        for i in 1:S, j in 1:S
            logξ[i, j, t] = (
                logα[i, t] + logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1]
            )
        end
        logsumξ = logsumexp(@view logξ[:, :, t])
        for i in 1:S, j in 1:S
            logξ[i, j, t] -= logsumξ
        end
    end

    logL = logsumexp(@view logα[:, T])

    return logL
end
