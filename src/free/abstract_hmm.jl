"""
    AbstractHiddenMarkovModel

Interface for Hidden Markov Models with arbitrary emissions.

# Required methods

- [`nb_states(hmm, par)`](@ref)
- [`initial_distribution(hmm, par)`](@ref)
- [`transition_matrix(hmm, par)`](@ref)
- [`emission_distribution(hmm, s, par)`](@ref)

# Optional methods

- [`log_initial_distribution(hmm, par)`](@ref)
- [`log_transition_matrix(hmm, par)`](@ref)

# Compatible with

- [`rand(rng, hmm, T, par)`](@ref)
- [`logdensityof(hmm, obs_sequence, par; safe)`](@ref)
- [`infer_current_state(hmm, obs_sequence, par; safe)`](@ref)
"""
abstract type AbstractHiddenMarkovModel end

"""
    AbstractHMM

Alias for [`AbstractHiddenMarkovModel`](@ref).
"""
const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Access

"""
    nb_states(hmm::AbstractHMM, par)

Return the number of states for `hmm` with parameters `par`.
"""
nb_states(hmm::H, par) where {H<:AbstractHMM} = error("Not implemented for type $H")

"""
    initial_distribution(hmm::AbstractHMM, par)

Return the vector of initial state probabilities for `hmm` with parameters `par`.
"""
function initial_distribution(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

"""
    log_initial_distribution(hmm::AbstractHMM, par)

Return the vector of initial state probabilities _in log scale_ for `hmm` with parameters `par`.
"""
function log_initial_distribution(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

"""
    transition_matrix(hmm::AbstractHMM, par)

Return the state transition matrix for `hmm` with parameters `par`.
"""
function transition_matrix(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

"""
    log_transition_matrix(hmm::AbstractHMM, par)

Return the state transition matrix _in log scale_ for `hmm` with parameters `par`.
"""
function log_transition_matrix(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

"""
    emission_distribution(hmm::AbstractHMM, s, par)

Return the distribution of the emissions in state `s` for `hmm` with parameters `par`.

The return object must be sampleable and implement [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl).
"""
function emission_distribution(hmm::H, s, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end
