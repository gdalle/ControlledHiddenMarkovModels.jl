abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

nb_states(hmm::AbstractControlledHMM) = error("Not implemented.")

initial_distribution(hmm::AbstractControlledHMM) = error("Not implemented.")

function transition_matrix(hmm::AbstractControlledHMM, control, parameters)
    return error("Not implemented.")
end

function transition_matrix!(
    P::AbstractMatrix, hmm::AbstractControlledHMM, control, parameters
)
    P .= transition_matrix(hmm, control, parameters)
    return P
end

function emission_parameters(hmm::AbstractControlledHMM, control, parameters)
    return error("Not implemented.")
end

function emission_parameters!(
    θ::AbstractMatrix, hmm::AbstractControlledHMM, control, parameters
)
    θ .= emission_parameters(hmm, control, parameters)
    return θ
end

function emission_from_parameters(hmm::AbstractControlledHMM, θ::AbstractVector)
    return error("Not implemented.")
end
