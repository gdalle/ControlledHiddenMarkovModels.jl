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
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(em[s], oₜ)
        end
    end
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
    return nothing
end

function compute_obs_density(hmm::AbstractHMM, obs_sequence::AbstractVector)
    T, S = length(obs_sequence), nb_states(hmm)
    em = emissions(hmm)
    o₁ = obs_sequence[1]
    test_density_value = densityof(em[1], o₁)
    obs_density = Matrix{typeof(test_density_value)}(undef, S, T)
    update_obs_density!(obs_density, hmm, obs_sequence)
    return obs_density
end
