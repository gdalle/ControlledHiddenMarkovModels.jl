abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

## Access

function nb_states(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function initial_distribution(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function log_initial_distribution(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function transition_matrix!(P, hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function log_transition_matrix!(logP, hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function transition_matrix(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function log_transition_matrix(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function emission_parameters!(θ, hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function emission_parameters(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

function emission_distribution(hmm::H, s, θ) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end
