function Base.rand(rng::AbstractRNG, hmm::HMM, T::Integer)
    states = rand(rng, transitions(hmm), T)
    observations = [rand(rng, emission(hmm, states[t])) for t = 1:T]
    return states, observations
end

Base.rand(hmm::HMM, T::Integer) = rand(GLOBAL_RNG, hmm, T)
