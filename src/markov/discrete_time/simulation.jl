function Base.rand(rng::AbstractRNG, mc::DiscreteMarkovChain, T::Integer)
    states = Vector{Int}(undef, T)
    states[1] = rand(rng, Categorical(mc.π0))
    transitions = [Categorical(mc.P[s, :]) for s in 1:nb_states(mc)]
    for t in 2:T
        states[t] = rand(rng, transitions[states[t - 1]])
    end
    return states
end

function Base.rand(rng::AbstractRNG, mc::DiscreteMarkovChainPrior)
    return error("not implemented")
end

Base.rand(mc::DiscreteMarkovChain, T::Integer) = rand(GLOBAL_RNG, mc, T)
Base.rand(prior::DiscreteMarkovChainPrior) = rand(GLOBAL_RNG, prior)
