"""
    rand([rng,] mc::DiscreteMarkovChain, T)

Simulate `mc` during `T` time steps.
"""
function Base.rand(rng::AbstractRNG, mc::DiscreteMarkovChain, T::Integer; check_args=false)
    states = Vector{Int}(undef, T)
    transitions = [Categorical(mc.P[s, :]; check_args=check_args) for s in 1:nb_states(mc)]
    states[1] = rand(rng, Categorical(mc.π0; check_args=check_args))
    for t in 2:T
        states[t] = rand(rng, transitions[states[t - 1]])
    end
    return states
end

"""
    rand([rng,] prior::DiscreteMarkovChainPrior)

Sample a [`DiscreteMarkovChain`](@ref) from `prior`.
"""
function Base.rand(rng::AbstractRNG, prior::DiscreteMarkovChainPrior; check_args=false)
    π0 = rand(rng, Dirichlet(prior.π0_α; check_args=check_args))
    P = reduce(
        vcat, rand(rng, Dirichlet(view(prior.P_α, s, :); check_args=check_args)) for s in 1:S
    )
    return DiscreteMarkovChain(π0, P)
end

function Base.rand(mc::DiscreteMarkovChain, T::Integer; kwargs...)
    return rand(GLOBAL_RNG, mc, T; kwargs...)
end

Base.rand(prior::DiscreteMarkovChainPrior; kwargs...) = rand(GLOBAL_RNG, prior; kwargs...)
