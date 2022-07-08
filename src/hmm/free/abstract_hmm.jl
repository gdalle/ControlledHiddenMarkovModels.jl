"""
    AbstractHiddenMarkovModel

Hidden Markov Model with arbitrary emissions.

# Required fields
- `p0::Vector{<:Real}`: initial distribution
- `P::Matrix{<:Real}`
- `emissions::Vector`: one emission distribution per state.
"""
abstract type AbstractHiddenMarkovModel end

const AbstractHMM = AbstractHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

## Access

nb_states(hmm::AbstractHMM) = length(hmm.p0)
initial_distribution(hmm::AbstractHMM) = hmm.p0
transition_matrix(hmm::AbstractHMM) = hmm.P
emissions(hmm::AbstractHMM) = hmm.emissions

function fit_emission_from_multiple_sequences(
    hmm::H, i::Integer, xs, ws
) where {H<:AbstractHMM}
    return error(
        "The method `fit_emission_from_multiple_sequences(hmm, i, xs, ws)` is not implemented for type $H.
        It is required for the Baum-Welch algorithm",
    )
end
