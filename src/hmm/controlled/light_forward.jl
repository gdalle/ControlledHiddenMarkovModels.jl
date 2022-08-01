function light_forward(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix{R},
    ps,
    st,
) where {R}
    S = nb_states(hmm)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm)
    P_all, θ_all = transition_matrix_and_emission_parameters(hmm, control_matrix, ps, st)
    # Initialization
    y₁ = obs_sequence[1]
    θ₁ = @view θ_all[:, :, 1]
    α = @views [p0[i] * densityof(emission_from_parameters(hmm, θ₁[:, i]), y₁) for i in 1:S]
    α_sum_inv = inv(sum(α))
    α .*= α_sum_inv
    logL = -log(α_sum_inv)
    # Recursion
    α_tmp = similar(α)
    @inbounds for t in 1:(T - 1)
        @info "Within forward" α
        Pₜ = @view P_all[:, :, t]
        yₜ₊₁ = obs_sequence[t + 1]
        θₜ₊₁ = @view θ_all[:, :, t + 1]
        @inbounds for j in 1:S
            emjₜ₊₁ = @views emission_from_parameters(hmm, θₜ₊₁[:, j])
            α_tmp[j] = sum(α[i] * Pₜ[i, j] for i in 1:S) * densityof(emjₜ₊₁, yₜ₊₁)
        end
        α .= α_tmp
        α_sum_inv = inv(sum(α))
        α .*= α_sum_inv
        logL -= log(α_sum_inv)
    end
    return α, logL
end

function light_logforward(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix{R},
    ps,
    st,
) where {R}
    S = nb_states(hmm)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm)
    P_all, θ_all = transition_matrix_and_emission_parameters(hmm, control_matrix, ps, st)
    # Initialization
    y₁ = obs_sequence[1]
    θ₁ = @view θ_all[:, :, 1]
    logα = @views [
        log(p0[i]) + logdensityof(emission_from_parameters(hmm, θ₁[:, i]), y₁) for i in 1:S
    ]
    # Recursion
    logα_tmp = similar(logα)
    @inbounds for t in 1:(T - 1)
        Pₜ = @view P_all[:, :, t]
        yₜ₊₁ = obs_sequence[t + 1]
        θₜ₊₁ = @view θ_all[:, :, t + 1]
        @inbounds for j in 1:S
            emjₜ₊₁ = @views emission_from_parameters(hmm, θₜ₊₁[:, j])
            logα_tmp[j] =
                logsumexp(logα[i] + log(Pₜ[i, j]) for i in 1:S) + logdensityof(emjₜ₊₁, yₜ₊₁)
        end
        logα .= logα_tmp
    end
    logL = logsumexp(logα)
    return exp.(logα .- logL), logL
end
