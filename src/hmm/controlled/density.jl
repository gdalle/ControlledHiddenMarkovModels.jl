function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractMatrix,
    args...,
)
    S = nb_states(hmm)
    T = length(obs_sequence)

    p0 = initial_distribution(hmm)
    P_all, θ_all = transition_matrix_and_emission_parameters(hmm, control_sequence, args...)

    y₁ = obs_sequence[1]
    θ₁ = view(θ_all, :, :, 1)
    α = [p0[i] * densityof(emission_from_parameters(hmm, view(θ₁, :, i)), y₁) for i in 1:S]
    α_sum_inv = inv(sum(α))
    α .*= α_sum_inv
    logL = -log(α_sum_inv)

    α_tmp = copy(α)

    for t in 1:(T - 1)
        Pₜ = view(P_all, :, :, t)
        yₜ₊₁ = obs_sequence[t + 1]
        θₜ₊₁ = view(θ_all, :, :, t + 1)
        for j in 1:S
            emjₜ₊₁ = emission_from_parameters(hmm, view(θₜ₊₁, :, j))
            α_tmp[j] = sum(α[i] * Pₜ[i, j] for i in 1:S) * densityof(emjₜ₊₁, yₜ₊₁)
        end
        α .= α_tmp
        α_sum_inv = inv(sum(α))
        α .*= α_sum_inv
        logL -= log(α_sum_inv)
    end

    return logL
end
