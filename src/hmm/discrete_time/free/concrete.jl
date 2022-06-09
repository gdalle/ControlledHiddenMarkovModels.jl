"""
    HiddenMarkovModel{Tr,Em}

Hidden Markov Model with arbitrary transition model (must be a discrete Markov chain) and emission distributions.

# Fields
- `transitions::Tr`: state evolution process.
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
initial_distribution(hmm::HMM) = initial_distribution(hmm.transitions)
transition_matrix(hmm::HMM, args...) = transition_matrix(hmm.transitions, args...)
emission_distribution(hmm::HMM, i::Integer, args...) = hmm.emissions[i]
