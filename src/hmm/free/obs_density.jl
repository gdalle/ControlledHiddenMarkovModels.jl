"""
    update_obs_density!(obs_density, hmm, obs_sequence)

Set `obs_density[s, t]` to the likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix{R}, hmm::AbstractHMM, obs_sequence::AbstractVector
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    em = emissions(hmm)
    for t in 1:T
        yₜ = obs_sequence[t]
        for i in 1:S
            obs_density[i, t] = densityof(em[i], yₜ)
        end
    end
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
end

function compute_obs_density(hmm::AbstractHMM, obs_sequence::AbstractVector)
    T, S = length(obs_sequence), nb_states(hmm)
    em = emissions(hmm)
    obs_density = [densityof(em[i], obs_sequence[t]) for i in 1:S, t in 1:T]
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
    return obs_density
end
