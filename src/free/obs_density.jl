"""
    update_obs_density!(obs_density, obs_sequence, hmm::AbstractHMM, par)

Update the values `obs_density[s, t]` using the emission density of `hmm` with parameters `par` applied to `obs_sequence[t]`.
"""
function update_obs_density!(
    obs_density::AbstractMatrix, obs_sequence::AbstractVector, hmm::AbstractHMM, par
)
    T, S = length(obs_sequence), nb_states(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(emissions[s], oₜ)
        end
    end
    return nothing
end

"""
    update_obs_logdensity!(obs_logdensity, obs_sequence, hmm::AbstractHMM, par)

Update the values `obs_logdensity[s, t]` using the emission log-density of `hmm` with parameters `par` applied to `obs_sequence[t]`.
"""
function update_obs_logdensity!(
    obs_logdensity::AbstractMatrix, obs_sequence::AbstractVector, hmm::AbstractHMM, par
)
    T, S = length(obs_sequence), nb_states(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_logdensity[s, t] = logdensityof(emissions[s], oₜ)
        end
    end
    return nothing
end

"""
    initialize_obs_density(obs_sequence, hmm, par)

Create a new observation density matrix and apply [`update_obs_density!`](@ref).
"""
function initialize_obs_density(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    T, S = length(obs_sequence), nb_states(hmm, par)
    test_density_value = logdensityof(emission_distribution(hmm, 1, par), obs_sequence[1])
    obs_density = Matrix{typeof(test_density_value)}(undef, S, T)
    update_obs_density!(obs_density, obs_sequence, hmm, par)
    return obs_density
end

"""
    initialize_obs_logdensity(obs_sequence, hmm, par)

Create a new observation log-density matrix and apply [`update_obs_logdensity!`](@ref).
"""
function initialize_obs_logdensity(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    T, S = length(obs_sequence), nb_states(hmm, par)
    test_logdensity_value = logdensityof(
        emission_distribution(hmm, 1, par), obs_sequence[1]
    )
    obs_logdensity = Matrix{typeof(test_logdensity_value)}(undef, S, T)
    update_obs_logdensity!(obs_logdensity, obs_sequence, hmm, par)
    return obs_logdensity
end
