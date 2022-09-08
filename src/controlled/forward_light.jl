function light_forward(
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    hmm::AbstractControlledHMM,
    par;
    safe=0,
)
    if safe == 0
        return light_forward_nolog(obs_sequence, control_sequence, hmm, par)
    elseif safe == 1
        return light_forward_log(obs_sequence, control_sequence, hmm, par)
    elseif safe == 2
        return light_forward_doublelog(obs_sequence, control_sequence, hmm, par)
    end
end

"""
    light_forward_nolog(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward_nolog(
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    hmm::AbstractControlledHMM,
    par,
)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    c₁ = control_sequence[1]
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, c₁, par)
    θ = emission_parameters(hmm, c₁, par)

    # Initialization
    o₁ = obs_sequence[1]
    obs_density = [densityof(emission_distribution(hmm, s, θ), o₁) for s in 1:S]
    α = p0 .* obs_density
    c = inv(sum(α))
    α .*= c
    logL = -log(c)

    # Recursion
    α_tmp = similar(α)
    for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, par)
        for s in 1:S
            obs_density[s] = densityof(emission_distribution(hmm, s, θ), oₜ₊₁)
        end
        mul!(α_tmp, P', α)
        transition_matrix!(P, hmm, cₜ₊₁, par)
        α_tmp .*= obs_density
        c = inv(sum(α_tmp))
        α_tmp .*= c
        logL -= log(c)
        α .= α_tmp
    end

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, float(logL)
end

"""
    light_forward_log(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass _partly in log scale_ with minimal storage requirements.
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
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, c₁, par)
    θ = emission_parameters(hmm, c₁, par)

    # Initialization
    o₁ = obs_sequence[1]
    obs_logdensity = [logdensityof(emission_distribution(hmm, s, θ), o₁) for s in 1:S]
    α = p0 .* exp.(obs_logdensity)
    c = inv(sum(α))
    α .*= c
    logL = -log(c)

    # Recursion
    α_tmp = similar(α)
    for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, par)
        for s in 1:S
            obs_logdensity[s] = logdensityof(emission_distribution(hmm, s, θ), oₜ₊₁)
        end
        mul!(α_tmp, P', α)
        transition_matrix!(P, hmm, cₜ₊₁, par)
        α_tmp .*= exp.(obs_logdensity)
        c = inv(sum(α_tmp))
        α_tmp .*= c
        logL -= log(c)
        α .= α_tmp
    end

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, float(logL)
end

"""
    light_forward_doublelog(obs_sequence, control_sequence, hmm::AbstractControlledHMM, par)

Perform a lightweight forward pass _fully in log scale_ with minimal storage requirements.
"""
function light_forward_doublelog(
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
