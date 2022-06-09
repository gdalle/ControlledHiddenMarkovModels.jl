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
    HMMPrior

Alias for [`HiddenMarkovModelPrior`](@ref).
"""
const HMMPrior = HiddenMarkovModelPrior
