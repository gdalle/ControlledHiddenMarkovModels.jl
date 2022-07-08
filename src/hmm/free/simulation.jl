"""
    rand(rng, hmm::AbstractHMM, T[, control_sequence])

Sample a sequence of states of length `T` and the associated sequence of observations.
"""
function Base.rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer; check_args=false)
    p0 = initial_distribution(hmm)
    P = transition_matrix(hmm)
    em = emissions(hmm)
    state_sequence = rand(rng, MarkovChain(p0, P), T; check_args=check_args)
    obs_sequence = [rand(rng, em[state_sequence[t]]) for t in 1:T]
    return state_sequence, obs_sequence
end

function Base.rand(hmm::AbstractHMM, T::Integer, check_args=false)
    return rand(GLOBAL_RNG, hmm, T; check_args=check_args)
end
