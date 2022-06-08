"""
    rand([rng,] mc::AbstractControlledDiscreteMarkovChain, controls)

Simulate `mc` based on a vector of controls.
"""
function Base.rand(
    rng::AbstractRNG,
    mc::AbstractControlledDiscreteMarkovChain,
    controls::AbstractVector;
    check_args=false,
)
    T = length(controls)
    p0 = initial_distribution(mc)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 2:T
        u, s = controls[t - 1], state_sequence[t - 1]
        P = transition_matrix(mc, u)
        state_sequence[t] = rand(rng, Categorical(P[s, :]; check_args=check_args))
    end
    return state_sequence
end

function Base.rand(
    mc::AbstractControlledDiscreteMarkovChain, controls::AbstractVector; kwargs...
)
    return rand(GLOBAL_RNG, mc, controls; kwargs...)
end
