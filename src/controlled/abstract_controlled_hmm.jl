"""
    AbstractControlledHiddenMarkovModel

Interface for Hidden Markov Models with arbitrary emissions and exogenous control variables.

# Required methods

- [`nb_states(hmm, par)`](@ref)
- [`initial_distribution(hmm, par)`](@ref)
- [`transition_matrix(hmm, control, par)`](@ref)
- [`emission_parameters(hmm, control, par)`](@ref)
- [`emission_distribution(hmm, s, θ)`](@ref)

# Compatible with

- [`rand(rng, hmm, control_sequence, par)`](@ref)
- [`logdensityof(hmm, obs_sequence, control_sequence, par; safe)`](@ref)
- [`infer_current_state(hmm, obs_sequence, control_sequence, par; safe)`](@ref)
"""
abstract type AbstractControlledHiddenMarkovModel end

"""
    AbstractControlledHMM

Alias for [`AbstractControlledHiddenMarkovModel`](@ref).
"""
const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

## Access

"""
    nb_states(hmm::AbstractControlledHMM, par)

Return the number of states for `hmm` with parameters `par`.
"""
function nb_states(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    initial_distribution(hmm::AbstractControlledHMM, par)

Return the vector of initial state probabilities for `hmm` with parameters `par`.
"""
function initial_distribution(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    transition_matrix(hmm::AbstractControlledHMM, control, par)

Compute the state transition matrix for `hmm` with control `control` and parameters `par`.
"""
function transition_matrix(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    emission_parameters(hmm::AbstractControlledHMM, control, par)

Compute the parameters of all emission distributions for `hmm` with control `control` and parameters `par`.
"""
function emission_parameters(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    emission_distribution(hmm::AbstractControlledHMM, s, θ)

Compute the emission distribution in state `s` for `hmm` with emission parameters `θ`.
Note that `θ` was computed using [`emission_parameters(hmm, control, par)`](@ref).

The object returned must be sampleable and implement [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl).
"""
function emission_distribution(hmm::H, s, θ) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

## Optional

"""
    log_initial_distribution(hmm::AbstractControlledHMM, par)

Return the vector of initial state probabilities _in log scale_ for `hmm` with parameters `par`.
"""
function log_initial_distribution(hmm::H, par) where {H<:AbstractControlledHMM}
    return log.(initial_distribution(hmm, par))
end

"""
    log_transition_matrix(hmm::AbstractControlledHMM, control, par)

Compute the state transition matrix _in log scale_ for `hmm` with control `control` and parameters `par`.
"""
function log_transition_matrix(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return log.(transition_matrix(hmm, control, par))
end
