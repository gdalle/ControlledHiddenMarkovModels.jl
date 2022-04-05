"""
    HiddenMarkovModel{T,E}

Hidden Markov Model with arbitrary transition model (must be a discrete Markov chain) and emission distributions.

# Fields
- `transitions::T`: state evolution process.
- `emissions::Vector{E}`: one emission distribution per state.
"""
struct HiddenMarkovModel{T,E}
    transitions::T
    emissions::Vector{E}
end

"""
    HiddenMarkovModelPrior{TP,EP}

Prior for a [`HiddenMarkovModel`](@ref).

# Fields
- `transitions_prior::T`: prior on the transition structure.
- `emissions_prior::Vector{E}`: one prior per state emission distribution.
"""
struct HiddenMarkovModelPrior{TP,EP}
    transitions_prior::TP
    emissions_prior::Vector{EP}
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
