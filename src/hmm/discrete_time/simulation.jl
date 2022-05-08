"""
    rand([rng,] hmm::HMM, T)

Sample a sequence of states of length `T` and the associated sequence of observations.
"""
function Base.rand(rng::AbstractRNG, hmm::HMM, T::Integer)
    state_sequence = rand(rng, transitions(hmm), T)
    obs_sequence = [rand(rng, emission(hmm, state_sequence[t])) for t = 1:T]
    return state_sequence, obs_sequence
end

Base.rand(hmm::HMM, T::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, T; kwargs...)
