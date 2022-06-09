"""
    rand([rng,] mc::AbstractDiscreteMarkovChain, T[, control_sequence, ps, st])

Simulate `mc` for `T` time steps.
"""
function Base.rand(
    rng::AbstractRNG,
    mc::AbstractDiscreteMarkovChain,
    T::Integer,
    control_sequence::AbstractVector=Fill(nothing, T),
    ps=nothing,
    st=nothing;
    check_args=false,
)
    T = length(control_sequence)
    p0 = initial_distribution(mc)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        u, s = control_sequence[t], state_sequence[t]
        P = transition_matrix(mc, u, ps, st)
        state_sequence[t + 1] = rand(rng, Categorical(P[s, :]; check_args=check_args))
    end
    return state_sequence
end
