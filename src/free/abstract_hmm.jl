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

transition_matrix(hmm::H, par) where {H<:AbstractHMM} = error("Not implemented for type $H")

function emission_distribution(hmm::H, s::Integer, par) where {H<:AbstractHMM}
    return error("Not implemented for type $H")
end
