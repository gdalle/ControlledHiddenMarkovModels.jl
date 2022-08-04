function light_forward(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    parameters,
)
    S = nb_states(hmm)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm)

    c₁ = control_sequence[1]
    P = transition_matrix(hmm, c₁, parameters)
    θ = emission_parameters(hmm, c₁, parameters)

    # Initialization
    o₁ = obs_sequence[1]
    α = [p0[s] * densityof(emission_from_parameters(hmm, θ, s), o₁) for s in 1:S]
    α_sum_inv = inv(sum(α))
    α .*= α_sum_inv
    logL = -log(α_sum_inv)

    # Recursion
    α_tmp = similar(α)
    @inbounds for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, parameters)
        oₜ₊₁ = obs_sequence[t + 1]
        @inbounds for j in 1:S
            emjₜ₊₁ = emission_from_parameters(hmm, θ, j)
            α_tmp[j] = sum(α[i] * P[i, j] for i in 1:S) * densityof(emjₜ₊₁, oₜ₊₁)
        end
        transition_matrix!(P, hmm, cₜ₊₁, parameters)
        α .= α_tmp
        α_sum_inv = inv(sum(α))
        α .*= α_sum_inv
        logL -= log(α_sum_inv)
    end

    @assert !isnan(logL)
    @assert !any(isnan, α)

    return α, logL
end

function light_logforward(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    parameters,
)
    S = nb_states(hmm)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm)

    c₁ = control_sequence[1]
    P = transition_matrix(hmm, c₁, parameters)
    θ = emission_parameters(hmm, c₁, parameters)

    # Initialization
    o₁ = obs_sequence[1]
    logα = [log(p0[s]) + logdensityof(emission_from_parameters(hmm, θ, s), o₁) for s in 1:S]

    # Recursion
    logα_tmp = similar(logα)
    @inbounds for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, parameters)
        oₜ₊₁ = obs_sequence[t + 1]
        @inbounds for j in 1:S
            emjₜ₊₁ = emission_from_parameters(hmm, θ, j)
            logα_tmp[j] =
                logsumexp(logα[i] + log(P[i, j]) for i in 1:S) + logdensityof(emjₜ₊₁, oₜ₊₁)
        end
        transition_matrix!(P, hmm, cₜ₊₁, parameters)
        logα .= logα_tmp
    end

    logL = logsumexp(logα)
    α = exp.(logα .- logL)

    @assert !isnan(logL)
    @assert !any(isnan, α)

    return α, logL
end
