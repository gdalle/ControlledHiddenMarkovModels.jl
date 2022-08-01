function update_obs_density!(
    obs_density::AbstractMatrix{R},
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix,
    ps,
    st,
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    θ_all = emission_parameters(hmm, control_matrix, ps, st)
    for t in 1:T
        yₜ = obs_sequence[t]
        θₜ = @view θ_all[:, :, t]
        for i in 1:S
            emiₜ = @views emission_from_parameters(hmm, θₜ[:, i])
            obs_density[i, t] = densityof(emiₜ, yₜ)
        end
    end
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
end

function compute_obs_density(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix,
    ps,
    st,
)
    T, S = length(obs_sequence), nb_states(hmm)
    θ_all = emission_parameters(hmm, control_matrix, ps, st)
    obs_density = @views [
        densityof(emission_from_parameters(hmm, θ_all[:, i, t]), obs_sequence[t]) for
        i in 1:S, t in 1:T
    ]
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
    return obs_density
end
