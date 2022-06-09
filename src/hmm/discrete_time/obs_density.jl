## Likelihood of obs_sequence

"""
    update_obs_density!(obs_density, hmm, obs_sequence)

Set `obs_density[s, t]` to the likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix{R},
    hmm::AbstractHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector=Fill(nothing, length(obs_sequence)),
    ps=nothing,
    st=nothing,
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    for t in 1:T
        uₜ = control_sequence[t]
        yₜ = obs_sequence[t]
        for i in 1:S
            em_dist = emission_distribution(hmm, i, uₜ, ps, st)
            obs_density[i, t] = densityof(em_dist, yₜ)
        end
    end
    for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
end

function compute_obs_density(
    hmm::AbstractHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector=Fill(nothing, length(obs_sequence)),
    ps=nothing,
    st=nothing,
)
    T, S = length(obs_sequence), nb_states(hmm)
    obs_density = [
        densityof(
            emission_distribution(hmm, i, control_sequence[t], ps, st), obs_sequence[t]
        ) for i in 1:S, t in 1:T
    ]
    for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
    return obs_density
end
