"""
    update_obs_density!(obs_density, obs_sequence, hmm::AbstractHMM, par)

Update the values `obs_density[s, t]` using the emission density of `hmm` with parameters `par` applied to `obs_sequence[t]`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix, obs_sequence, hmm::AbstractHMM, par
)
    T, S = length(obs_sequence), nb_states(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(emissions[s], oₜ)
        end
    end
    @assert !any(isnan, obs_density)
    return nothing
end

"""
    initialize_obs_density(obs_sequence, hmm, par)

Create a new observation density matrix and apply [`update_obs_density!`](@ref).
"""
function initialize_obs_density(obs_sequence, hmm::AbstractHMM, par)
    T, S = length(obs_sequence), nb_states(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    obs_density = [densityof(emissions[s], obs_sequence[t]) for s in 1:S, t in 1:T]
    @assert !any(isnan, obs_density)
    return obs_density
end
