"""
    light_forward_log(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass _in log scale_ with minimal storage requirements.
"""
function light_forward_log(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    logα = [logp0[i] + logdensityof(emissions[i], o₁) for i in 1:S]

    # Recursion
    logα_tmp = similar(logα)
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        for j in 1:S
            logα_tmp[j] = logsumexp(logP[i, j] + logα[i] for i in 1:S)
            logα_tmp[j] += logdensityof(emissions[j], oₜ₊₁)
        end
        logα .= logα_tmp
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)

    return exp.(logα .- logsumexp(logα)), logL
end

"""
    forward_log!(logα, obs_logdensity, hmm::AbstractHMM, par)

Perform a forward pass _in log scale_ by mutating `logα`.
"""
function forward_log!(
    logα::AbstractMatrix, obs_logdensity::AbstractMatrix, hmm::AbstractHMM, par
)
    S, T = size(obs_logdensity)
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, par)
    @assert !any(isnan, logp0)
    @assert !any(isnan, logP)

    # Initialization
    @views logα[:, 1] .= logp0 .+ obs_logdensity[:, 1]

    # Recursion
    @views for t in 1:(T - 1)
        for j in 1:S
            logα[j, t + 1] = (
                logsumexp(logα[i, t] + logP[i, j] for i in 1:S) + obs_logdensity[j, t + 1]
            )
        end
    end
    @assert !any(isnan, logα)
    return nothing
end

"""
    backward_log!(logβ, obs_logdensity, hmm::AbstractHMM, par)

Perform a backward pass _in log scale_ by mutating `logβ`.
"""
function backward_log!(
    logβ::AbstractMatrix, obs_logdensity::AbstractMatrix, hmm::AbstractHMM, par
)
    S, T = size(obs_logdensity)
    logP = log_transition_matrix(hmm, par)
    @assert !any(isnan, logP)

    # Initialization
    @views logβ[:, T] .= zero(eltype(logβ))

    # Recursion
    @views for t in (T - 1):-1:1
        for i in 1:S
            logβ[i, t] = logsumexp(
                logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1] for j in 1:S
            )
        end
    end
    @assert !any(isnan, logβ)
    return nothing
end

"""
    forward_backward_log!(logα, logβ, logγ, logξ, obs_logdensity, hmm::AbstractHMM, par)

Apply the full forward-backward algorithm _in log scale_ by mutating `logα`, `logβ`, `logγ` and `logξ`.
"""
function forward_backward_log!(
    logα::AbstractMatrix,
    logβ::AbstractMatrix,
    logγ::AbstractMatrix,
    logξ::AbstractArray{<:Real,3},
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_logdensity)
    logP = log_transition_matrix(hmm, par)

    # Forward and backward pass
    forward_log!(logα, obs_logdensity, hmm, par)
    backward_log!(logβ, obs_logdensity, hmm, par)

    # State marginals
    logγ .= logα .+ logβ
    @views for t in 1:T
        logγ[:, t] .-= logsumexp(logγ[:, t])
    end

    # Transition marginals
    @views for t in 1:(T - 1)
        for j in 1:S
            for i in 1:S
                logξ[i, j, t] = (
                    logα[i, t] + logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1]
                )
            end
        end
        logξ[:, :, t] .-= logsumexp(logξ[:, :, t])
    end

    logL = logsumexp(logα)
    return float(logL)
end
