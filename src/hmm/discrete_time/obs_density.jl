## Likelihood of obs_sequence

"""
    update_obs_density!(obs_density, hmm, obs_sequence)

Set `obs_density[s, t]` to the likelihood of `hmm` emitting `obs_sequence[t]` if it were in state `s`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix{R}, hmm::HMM, obs_sequence::AbstractVector
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(get_emission(hmm, s), oₜ)
        end
    end
    for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
end

function compute_obs_density(
    hmm::HMM{Tr,Em}, obs_sequence::AbstractVector{O},
) where {Tr,Em,O}
    T, S = length(obs_sequence), nb_states(hmm)
    obs_density = [densityof(get_emission(hmm, s), obs_sequence[t]) for s = 1:S, t = 1:T]
    for t in 1:T
        if all(iszero_safe, view(obs_density, :, t))
            throw(OverflowError("Densities are too small for observations."))
        end
    end
    return obs_density
end
