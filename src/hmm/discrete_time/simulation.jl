"""
    rand([rng,] hmm::HMM, T)

Sample a sequence of states of length `T` and the associated sequence of observations.
"""
function Base.rand(
    rng::AbstractRNG, hmm::HMM{Tr}, T::Integer
) where {Tr<:AbstractDiscreteMarkovChain}
    state_sequence = rand(rng, get_transitions(hmm), T)
    obs_sequence = [rand(rng, get_emission(hmm, state_sequence[t])) for t in 1:T]
    return state_sequence, obs_sequence
end

function Base.rand(
    rng::AbstractRNG, hmm::HMM{Tr}, control_sequence::AbstractVector
) where {Tr<:AbstractControlledDiscreteMarkovChain}
    T = length(control_sequence)
    state_sequence = rand(rng, get_transitions(hmm), control_sequence)
    obs_sequence = [rand(rng, get_emission(hmm, state_sequence[t])) for t in 1:T]
    return state_sequence, obs_sequence
end
