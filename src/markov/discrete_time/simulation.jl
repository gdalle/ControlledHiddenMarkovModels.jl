"""
    rand([rng,] mc::AbstractDiscreteMarkovChain, T)

Simulate `mc` during `T` time steps.
"""
function Base.rand(
    rng::AbstractRNG, mc::AbstractDiscreteMarkovChain, T::Integer; check_args=false
)
    state_sequence = Vector{Int}(undef, T)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    transitions = [Categorical(P[s, :]; check_args=check_args) for s in 1:nb_states(mc)]
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 2:T
        state_sequence[t] = rand(rng, transitions[state_sequence[t - 1]])
    end
    return state_sequence
end

function Base.rand(mc::AbstractDiscreteMarkovChain, T::Integer; kwargs...)
    return rand(GLOBAL_RNG, mc, T; kwargs...)
end
