"""
    AbstractHiddenMarkovModel

Hidden Markov Model with arbitrary emissions.
"""
abstract type AbstractHiddenMarkovModel end

const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Access

nb_states(hmm::H, par) where {H<:AbstractHMM} = error("Not implemented for type $H")

function initial_distribution(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

function log_initial_distribution(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

function transition_matrix(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

function log_transition_matrix(hmm::H, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end

function emission_distribution(hmm::H, s, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end
