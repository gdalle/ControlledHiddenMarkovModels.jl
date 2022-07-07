abstract type AbstractHiddenMarkovModel end

const AbstractHMM = AbstractHiddenMarkovModel

nb_states(::AbstractHMM) = error("not implemented")
transitions(::AbstractHMM, args...) = error("not implemented")
emissions(::AbstractHMM, args...) = error("not implemented")

function initial_distribution(ahmm::AbstractHMM, args...)
    return initial_distribution(transitions(ahmm, args...))
end

function transition_matrix(ahmm::AbstractHMM, args...)
    return transition_matrix(transitions(ahmm, args...))
end

emissions(hmm::AbstractHMM, i::Integer, args...) = emissions(hmm, args...)[i]

function transitions_and_emissions(hmm::AbstractHMM, args...)
    return (transitions(hmm, args...), emissions(hmm, args...))
end
