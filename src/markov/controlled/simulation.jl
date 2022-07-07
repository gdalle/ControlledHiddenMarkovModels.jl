function Base.rand(
    rng::AbstractRNG,
    mc::ControlledMarkovChain,
    control_sequence::AbstractMatrix{<:Real},
    args...;
    check_args=false,
)
    p0 = initial_distribution(mc)
    P_all = transition_matrix(mc, control_sequence, args...)
    T = size(control_sequence, 2)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        iₜ = state_sequence[t]
        Pₜ = view(P_all, :, :, t)
        iₜ₊₁ = rand(rng, Categorical(view(Pₜ, iₜ, :); check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end
    return state_sequence
end

function Base.rand(
    mc::ControlledMarkovChain,
    control_sequence::AbstractMatrix{<:Real},
    args...;
    check_args=false,
)
    return rand(GLOBAL_RNG, mc, control_sequence, args...; check_args=check_args)
end