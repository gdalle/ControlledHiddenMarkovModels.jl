## Forward-backward

function forward_log!(
    logα::AbstractMatrix{R}, hmm::HMM, obs_logdensity::AbstractMatrix{R}
) where {R<:Real}
    S, T = size(obs_logdensity)
    logp0, logP = log.(initial_distribution(hmm)), log.(transition_matrix(hmm))

    # Initialization
    for i in 1:S
        logα[i, 1] = logp0[i] + obs_logdensity[i, 1]
    end

    # Recursion
    for t in 1:(T - 1)
        logα_prev = view(logα, :, t)
        for j in 1:S
            logα[j, t + 1] =
                logsumexpsum(logα_prev, view(logP, :, j)) + obs_logdensity[j, t + 1]
        end
    end

    # Overflow check
    for t in 1:T
        if all(isnan, view(logα, t, :))
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

    # Initialization
    for i in 1:S
        logβ[i, T] = zero(R)
    end

    # Recursion
    for t in (T - 1):-1:1
        obs_logdensity_next = view(obs_logdensity, :, t + 1)
        logβ_next = view(logβ, :, t + 1)
        for i in 1:S
            logβ[t, i] = logsumexpsum(view(logP, i, :), obs_logdensity_next, logβ_next)
        end
    end

    # Overflow check
    for t in 1:T
        if all(isnan, view(logβ, t, :))
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

    # State sufficient statistics
    for t in 1:T
        for i in 1:S
            logγ[i, t] = logα[i, t] + logβ[i, t]
        end
        logγ_sum = logsumexp(view(logγ, :, t))
        for i in 1:S
            logγ[i, t] -= logγ_sum
        end
    end

    # Transitions sufficient statistics
    for t in 1:(T - 1)
        for j in 1:S, i in 1:S
            logξ[i, j, t] = (
                logα[i, t] + logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1]
            )
        end
        logξ_sum = logsumexp(view(logξ, :, :, t))
        for j in 1:S, i in 1:S
            logξ[i, j, t] -= logξ_sum
        end
    end

    logL = logsumexp(view(logα, :, T))

    return logL
end
