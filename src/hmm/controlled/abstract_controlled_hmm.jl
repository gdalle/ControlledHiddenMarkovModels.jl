abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

nb_states(::AbstractControlledHMM) = error("Not implemented.")

initial_distribution(::AbstractControlledHMM) = error("Not implemented.")

transition_matrix(::AbstractControlledHMM, u, args...) = error("Not implemented.")

emission_parameters(::AbstractControlledHMM, u, args...) = error("Not implemented.")
emission_from_parameters(::AbstractControlledHMM, θ) = error("Not implemented.")

function transition_matrix_and_emission_parameters(hmm::AbstractControlledHMM, u, args...)
    return (transition_matrix(hmm, u, args...), emission_parameters(hmm, u, args...))
end
