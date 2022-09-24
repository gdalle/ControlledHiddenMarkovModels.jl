"""
    light_forward(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward(obs_sequence, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    obs_density = [densityof(emissions[s], o₁) for s in 1:S]
    α = p0 .* obs_density
    c = inv(sum(α))
    α = α .* c
    logL = -log(c)

    # Recursion
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        obs_density = [densityof(emissions[s], oₜ₊₁) for s in 1:S]
        α = (P' * α) .* obs_density
        c = inv(sum(α))
        α = α .* c
        logL -= log(c)
    end

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, logL
end

function light_forward_log(obs_sequence, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    obs_logdensity = [logdensityof(emissions[s], o₁) for s in 1:S]
    logα = logp0 .+ obs_logdensity

    # Recursion
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        obs_logdensity = [logdensityof(emissions[s], oₜ₊₁) for s in 1:S]
        logα = [
            logsumexp(logP[i, j] + logα[i] for i in 1:S) + obs_logdensity[j] for j in 1:S
        ]
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)
    return exp.(logα .- logL), logL
end
