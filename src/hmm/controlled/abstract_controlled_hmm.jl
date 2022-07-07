abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

nb_states(::AbstractControlledHMM) = error("not implemented")

initial_distribution(::AbstractControlledHMM) = error("not implemented")

transition_matrix(::AbstractControlledHMM, u, args...) = error("not implemented")

emission_parameters(::AbstractControlledHMM, u, args...) = error("not implemented")
emission_from_parameters(::AbstractControlledHMM, θ) = error("not implemented")

function transition_matrix_and_emission_parameters(hmm::AbstractControlledHMM, u, args...)
    return (transition_matrix(hmm, u, args...), emission_parameters(hmm, u, args...))
end
