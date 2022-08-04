
function light_forward(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    parameters,
)
    S = nb_states(hmm)
    T = length(obs_sequence)
    logp0 = log_initial_distribution(hmm)

    c₁ = control_sequence[1]
    logP = log_transition_matrix(hmm, c₁, parameters)
    θ = emission_parameters(hmm, c₁, parameters)

    # Initialization
    o₁ = obs_sequence[1]
    logα = [logp0[s] + logdensityof(emission_from_parameters(hmm, θ, s), o₁) for s in 1:S]

    # Recursion
    logα_tmp = similar(logα)
    @inbounds for t in 1:(T - 1)
        cₜ₊₁ = control_sequence[t + 1]
        oₜ₊₁ = obs_sequence[t + 1]
        emission_parameters!(θ, hmm, cₜ₊₁, parameters)
        @inbounds for j in 1:S
            emjₜ₊₁ = emission_from_parameters(hmm, θ, j)
            logα_tmp[j] = logsumexp_stream(eltype(logα), logα[i] + logP[i, j] for i in 1:S)
            logα_tmp[j] += logdensityof(emjₜ₊₁, oₜ₊₁)
        end
        log_transition_matrix!(logP, hmm, cₜ₊₁, parameters)
        logα .= logα_tmp
    end

    logL = logsumexp_stream(logα)
    α = exp.(logα .- logL)

    @assert !isnan(logL)
    @assert !any(isnan, α)

    return α, logL
end
