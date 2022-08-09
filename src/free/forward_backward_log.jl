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
