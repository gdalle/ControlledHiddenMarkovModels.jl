"""
    HiddenMarkovModel{Tr,Em}

Hidden Markov Model with arbitrary transition model (must be a discrete Markov chain) and emission distributions.

# Fields
- `transitions::Tr`: state evolution process.
- `emissions::Vector{Em}`: one emission distribution per state.
"""
struct HiddenMarkovModel{Tr,Em}
    transitions::Tr
    emissions::Vector{Em}
end

"""
    HiddenMarkovModelPrior{TrP,EmP}

Prior for a [`HiddenMarkovModel`](@ref).

# Fields
- `transitions_prior::TrP`: prior on the transition structure.
- `emissions_prior::Vector{EmP}`: one prior per state emission distribution.
"""
struct HiddenMarkovModelPrior{TrP,EmP}
    transitions_prior::TrP
    emissions_prior::Vector{EmP}
end

"""
    HMM

Alias for [`HiddenMarkovModel`](@ref).
"""
const HMM = HiddenMarkovModel

"""
    HMMPrior

Alias for [`HiddenMarkovModelPrior`](@ref).
"""
const HMMPrior = HiddenMarkovModelPrior

## Access

transitions(hmm::HMM) = hmm.transitions
initial_distribution(hmm::HMM) = initial_distribution(transitions(hmm))
transition_matrix(hmm::HMM) = transition_matrix(transitions(hmm))

emissions(hmm::HMM) = hmm.emissions
emission(hmm::HMM, s::Integer) = hmm.emissions[s]
nb_states(hmm::HMM) = length(emissions(hmm))
