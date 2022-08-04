abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

nb_states(hmm::AbstractControlledHMM) = error("Not implemented.")

## Log versions

log_initial_distribution(hmm::AbstractControlledHMM, parameters) = error("Not implemented.")

function log_transition_matrix(hmm::AbstractControlledHMM, control, parameters)
    return error("Not implemented.")
end

function log_transition_matrix!(
    logP::AbstractMatrix, hmm::AbstractControlledHMM, control, parameters
)
    logP .= log_transition_matrix(hmm, control, parameters)
    return logP
end

function emission_parameters(hmm::AbstractControlledHMM, control, parameters)
    return error("Not implemented.")
end

function emission_parameters!(
    θ::AbstractArray, hmm::AbstractControlledHMM, control, parameters
)
    θ .= emission_parameters(hmm, control, parameters)
    return θ
end

function emission_distribution(hmm::AbstractControlledHMM, θ::AbstractArray, s::Integer)
    return error("Not implemented.")
end

## Normal versions

function initial_distribution(hmm::AbstractControlledHMM, parameters)
    return exp.(log_initial_distribution(hmm, parameters))
end

function transition_matrix!(
    P::AbstractMatrix, hmm::AbstractControlledHMM, control, parameters
)
    log_transition_matrix!(P, hmm, control, parameters)
    P .= exp.(P)
    return P
end

function transition_matrix(hmm::AbstractControlledHMM, control, parameters)
    return exp.(log_transition_matrix(hmm, control, parameters))
end
