function Base.rand(
    rng::AbstractRNG,
    mc::AbstractControlledMarkovChain,
    control_sequence::AbstractVector,
    params;
    check_args=false,
)
    p0 = initial_distribution(mc)
    T = length(control_sequence)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    c₁ = control_sequence[1]
    P = transition_matrix(mc, c₁, params)
    @views for t in 1:(T - 1)
        sₜ = state_sequence[t]
        sₜ₊₁ = rand(rng, Categorical(P[sₜ, :]; check_args=check_args))
        state_sequence[t + 1] = sₜ₊₁
        cₜ₊₁ = control_sequence[t + 1]
        transition_matrix!(P, mc, cₜ₊₁, params)
    end
    return state_sequence
end

function Base.rand(
    mc::AbstractControlledMarkovChain,
    control_sequence::AbstractVector,
    params;
    check_args=false,
)
    return rand(GLOBAL_RNG, mc, control_sequence, params; check_args=check_args)
end
