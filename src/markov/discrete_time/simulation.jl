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
    state_sequence = Vector{Int}(undef, T)
    p0 = initial_distribution(mc)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        iₜ = state_sequence[t]
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(mc, uₜ, ps, st)
        iₜ₊₁ = rand(rng, Categorical(view(Pₜ, iₜ, :); check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end
    return state_sequence
end
