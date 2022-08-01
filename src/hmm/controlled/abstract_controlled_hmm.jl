abstract type AbstractControlledHiddenMarkovModel end

const AbstractControlledHMM = AbstractControlledHiddenMarkovModel

@inline DensityInterface.DensityKind(::AbstractControlledHMM) = HasDensity()

nb_states(::AbstractControlledHMM) = error("Not implemented.")

initial_distribution(::AbstractControlledHMM) = error("Not implemented.")

transition_matrix(::AbstractControlledHMM, u, ps, st) = error("Not implemented.")

emission_parameters(::AbstractControlledHMM, u, ps, st) = error("Not implemented.")

emission_from_parameters(::AbstractControlledHMM, θ) = error("Not implemented.")

function transition_matrix_and_emission_parameters(hmm::AbstractControlledHMM, u, ps, st)
    return (transition_matrix(hmm, u, ps, st), emission_parameters(hmm, u, ps, st))
end
