"""
    light_forward(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm, par)
    all_P = [transition_matrix(hmm, control_sequence[t], par) for t in 1:T]
    all_θ = [emission_parameters(hmm, control_sequence[t], par) for t in 1:T]
    obs_density = [
        densityof(emission_distribution(hmm, s, all_θ[t]), obs_sequence[t]) for s in 1:S,
        t in 1:T
    ]

    # Initialization
    α = p0 .* @view obs_density[:, 1]
    c = inv(sum(α))
    α = α .* c
    logL = -log(c)

    # Recursion
    for t in 1:(T - 1)
        α = (all_P[t]' * α) .* @view obs_density[:, t + 1]
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
    logp0 = log_initial_distribution(hmm, par)
    all_logP = [log_transition_matrix(hmm, control_sequence[t], par) for t in 1:T]
    all_θ = [emission_parameters(hmm, control_sequence[t], par) for t in 1:T]
    obs_logdensity = [
        logdensityof(emission_distribution(hmm, s, all_θ[t]), obs_sequence[t]) for s in 1:S,
        t in 1:T
    ]

    # Initialization
    logα = logp0 .+ @view obs_logdensity[:, 1]

    # Recursion
    for t in 1:(T - 1)
        logα = (
            dropdims(logsumexp(all_logP[t] .+ logα; dims=1); dims=1) .+
            @view obs_logdensity[:, t + 1]
        )
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)
    return exp.(logα .- logL), logL
end
