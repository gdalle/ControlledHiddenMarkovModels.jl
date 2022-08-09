"""
    light_forward_log(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass _in log scale_ with minimal storage requirements.
"""
function light_forward_log(
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    hmm::AbstractControlledHMM,
    par,
)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    c₁ = control_sequence[1]
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, c₁, par)
    θ = emission_parameters(hmm, c₁, par)

    # Initialization
    o₁ = obs_sequence[1]
    logα = [logp0[i] + logdensityof(emission_distribution(hmm, i, θ), o₁) for i in 1:S]

    # Recursion
    logα_tmp = similar(logα)
    for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, par)
        for j in 1:S
            logα_tmp[j] = logsumexp(logP[i, j] + logα[i] for i in 1:S)
            logα_tmp[j] += logdensityof(emission_distribution(hmm, j, θ), oₜ₊₁)
        end
        log_transition_matrix!(logP, hmm, cₜ₊₁, par)
        logα .= logα_tmp
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)

    return exp.(logα .- logsumexp(logα)), logL
end
