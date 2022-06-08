"""
    HiddenMarkovModel{Tr,Em}

Hidden Markov Model with arbitrary transition model (must be a discrete Markov chain) and emission distributions.

# Fields
- `transitions::Tr`: state evolution process.
- `emissions::Vector{Em}`: one emission distribution per state.
"""
struct HiddenMarkovModel{Tr<:AbstractDiscreteMarkovChain,Em}
    transitions::Tr
    emissions::Vector{Em}
end

"""
    HMM

Alias for [`HiddenMarkovModel`](@ref).
"""
const HMM = HiddenMarkovModel

## Access

get_transitions(hmm::HMM) = hmm.transitions
initial_distribution(hmm::HMM) = initial_distribution(get_transitions(hmm))
transition_matrix(hmm::HMM) = transition_matrix(get_transitions(hmm))

get_emissions(hmm::HMM) = hmm.emissions
get_emission(hmm::HMM, s::Integer) = hmm.emissions[s]
nb_states(hmm::HMM) = length(get_emissions(hmm))
