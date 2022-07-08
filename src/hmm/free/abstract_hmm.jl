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

emission_type(::Type{<:AbstractHMM}) = error("Not implemented.")

function fit_emission_from_multiple_sequences(hmm::Type{H}, xs, ws) where {H<:AbstractHMM}
    return error(
        "The method `fit_emission_from_multiple_sequences(H, xs, ws)` is not implemented for type $H.
        It is required for the Baum-Welch algorithm",
    )
end
