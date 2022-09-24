"""
    light_forward(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)

    # Initialization
    u₁ = control_sequence[1]
    o₁ = obs_sequence[1]
    p0 = initial_distribution(hmm, par)
    θ = emission_parameters(hmm, u₁, par)
    obs_density = [densityof(emission_distribution(hmm, s, θ), o₁) for s in 1:S]
    α = p0 .* obs_density
    c = inv(sum(α))
    α = α .* c
    logL = -log(c)

    # Recursion
    for t in 1:(T - 1)
        uₜ = control_sequence[t]
        uₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        P = transition_matrix(hmm, uₜ, par)
        θ = emission_parameters(hmm, uₜ₊₁, par)
        obs_density = [densityof(emission_distribution(hmm, s, θ), oₜ₊₁) for s in 1:S]

        α = (P' * α) .* obs_density
        c = inv(sum(α))
        α = α .* c
        logL -= log(c)
    end

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, logL
end

function light_forward_log(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)

    # Initialization
    u₁ = control_sequence[1]
    o₁ = obs_sequence[1]
    logp0 = log_initial_distribution(hmm, par)
    θ = emission_parameters(hmm, u₁, par)
    obs_logdensity = [logdensityof(emission_distribution(hmm, s, θ), o₁) for s in 1:S]
    logα = logp0 .+ obs_logdensity

    # Recursion
    for t in 1:(T - 1)
        uₜ = control_sequence[t]
        uₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        logP = log_transition_matrix(hmm, uₜ, par)
        θ = emission_parameters(hmm, uₜ₊₁, par)
        obs_logdensity = [logdensityof(emission_distribution(hmm, s, θ), oₜ₊₁) for s in 1:S]
        logα = [
            logsumexp(logP[i, j] .+ logα[i] for i in 1:S) + obs_logdensity[j] for j in 1:S
        ]
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)
    return exp.(logα .- logL), logL
end
