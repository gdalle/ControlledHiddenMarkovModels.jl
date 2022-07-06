abstract type AbstractHiddenMarkovModel end

const AbstractHMM = AbstractHiddenMarkovModel

nb_states(hmm::AbstractHMM) = error("not implemented")
initial_distribution(hmm::AbstractHMM, args...) = error("not implemented")
transition_matrix(hmm::AbstractHMM, args...) = error("not implemented")
emission_distribution(hmm::AbstractHMM, i::Integer, args...) = error("not implemented")

function transition_matrix_and_emission_distributions(hmm::AbstractHMM, i::Integer, args...)
    return error("not implemented")
end
