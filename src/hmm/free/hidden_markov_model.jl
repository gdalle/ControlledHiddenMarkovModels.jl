"""
    HiddenMarkovModel

Hidden Markov Model with arbitrary emissions.

# Fields
- `transitions::TR`: underlying Markov chain
- `emissions::Vector{Em}`: one emission distribution per state.
"""
struct HiddenMarkovModel{Tr,Em} <: AbstractHMM
    transitions::Tr
    emissions::Vector{Em}
end

"""
    HMM

Alias for [`HiddenMarkovModel`](@ref).
"""
const HMM = HiddenMarkovModel

## Access

nb_states(hmm::HMM) = length(hmm.emissions)
transitions(hmm::HMM) = hmm.transitions
emissions(hmm::HMM) = hmm.emissions
