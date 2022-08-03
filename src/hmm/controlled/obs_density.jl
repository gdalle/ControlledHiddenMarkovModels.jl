function update_obs_density!(
    obs_density::AbstractMatrix{R},
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    params,
) where {R<:Real}
    T, S = length(obs_sequence), nb_states(hmm)
    c₁ = control_sequence[1]
    θ = emission_parameters(hmm, c₁, params)
    for t in 1:T
        oₜ = obs_sequence[t]
        @views for s in 1:S
            emsₜ = emission_from_parameters(hmm, θ[:, s])
            obs_density[s, t] = densityof(emsₜ, oₜ)
        end
        if t < T
            cₜ₊₁ = control_sequence[t + 1]
            emission_parameters!(θ, hmm, cₜ₊₁, params)
        end
    end
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
end

function compute_obs_density(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    params,
)
    T, S = length(obs_sequence), nb_states(hmm)
    c₁ = control_sequence[1]
    o₁ = obs_sequence[1]
    θ = emission_parameters(hmm, c₁, params)
    test_density_value = @views densityof(emission_from_parameters(hmm, θ[:, 1]), o₁)
    obs_density = Matrix{typeof(test_density_value)}(undef, S, T)
    update_obs_density!(obs_density, hmm, obs_sequence, control_sequence, params)
    return obs_density
end
