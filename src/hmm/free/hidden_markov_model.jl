"""
    HiddenMarkovModel

Hidden Markov Model with arbitrary emissions.

# Fields
- `emissions::Vector{Em}`: one emission distribution per state.
"""
struct HiddenMarkovModel{R1,R2,Em} <: AbstractHMM
    p0::Vector{R1}
    P::Matrix{R2}
    emissions::Vector{Em}
end

"""
    HMM

Alias for [`HiddenMarkovModel`](@ref).
"""
const HMM = HiddenMarkovModel

## Access

nb_states(hmm::HMM) = length(hmm.p0)
initial_distribution(hmm::HMM) = hmm.p0
transition_matrix(hmm::HMM) = hmm.P
emissions(hmm::HMM) = hmm.emissions
