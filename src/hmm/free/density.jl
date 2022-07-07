function fast_forward(
    hmm::HMM,
    obs_sequence::AbstractVector,
)
    S = nb_states(hmm)
    T = length(obs_sequence)

    p0 = initial_distribution(hmm)
    P = transition_matrix(hmm)
    em = emissions(hmm)

    α = [p0[i] * densityof(em[i], y₁) for i in 1:S]
    α_sum_inv = inv(sum(α))
    α .*= α_sum_inv
    logL = -log(α_sum_inv)

    α_tmp = similar(α)

    @inbounds for t in 1:(T - 1)
        yₜ₊₁ = obs_sequence[t + 1]
        @inbounds for j in 1:S
            α_tmp[j] = sum(α[i] * P[i, j] for i in 1:S) * densityof(em[j], yₜ₊₁)
        end
        α .= α_tmp
        α_sum_inv = inv(sum(α))
        α .*= α_sum_inv
        logL -= log(α_sum_inv)
    end

    return α, logL
end

function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractMatrix,
    args...,
)
    α, logL = fast_forward(hmm, obs_sequence, control_sequence, args...)
    return logL
end
