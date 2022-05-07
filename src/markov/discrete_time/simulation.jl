"""
    rand([rng,] mc::DiscreteMarkovChain, T)

Simulate `mc` during `T` time steps.
"""
function Base.rand(rng::AbstractRNG, mc::DiscreteMarkovChain, T::Integer)
    states = Vector{Int}(undef, T)
    transitions = [Categorical(mc.P[s, :]) for s in 1:nb_states(mc)]
    states[1] = rand(rng, Categorical(mc.π0))
    for t in 2:T
        states[t] = rand(rng, transitions[states[t - 1]])
    end
    return states
end

"""
    rand([rng,] prior::DiscreteMarkovChainPrior)

Sample a [`DiscreteMarkovChain`](@ref) from `prior`.
"""
function Base.rand(rng::AbstractRNG, prior::DiscreteMarkovChainPrior)
    return error("not implemented")
end

Base.rand(mc::DiscreteMarkovChain, T::Integer) = rand(GLOBAL_RNG, mc, T)
Base.rand(prior::DiscreteMarkovChainPrior) = rand(GLOBAL_RNG, prior)
