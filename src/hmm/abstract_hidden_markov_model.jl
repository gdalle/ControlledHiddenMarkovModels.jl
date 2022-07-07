abstract type AbstractHiddenMarkovModel end
abstract type AbstractControlledHiddenMarkovModel <: AbstractHiddenMarkovModel end

const AbstractHMM = AbstractHiddenMarkovModel
const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractHMM) = HasDensity()

nb_states(::AbstractHMM) = error("not implemented")

initial_distribution(::AbstractHMM) = error("not implemented")

transition_matrix(::AbstractHMM) = error("not implemented")
transition_matrix(::AbstractControlledHMM, u, args...) = error("not implemented")

emissions(::AbstractHMM) = error("not implemented")

emission_parameters(::AbstractControlledHMM, u, args...) = error("not implemented")
emission_from_parameters(::AbstractControlledHMM, θ) = error("not implemented")

function transition_matrix_and_emission_parameters(hmm::AbstractControlledHMM, u, args...)
    return (transition_matrix(hmm, u, args...), emission_parameters(hmm, u, args...))
end
