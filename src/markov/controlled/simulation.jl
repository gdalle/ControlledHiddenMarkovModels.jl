function Base.rand(
    rng::AbstractRNG,
    mc::AbstractControlledMarkovChain,
    control_sequence::AbstractVector,
    parameters;
    check_args=false,
)
    p0 = initial_distribution(mc, parameters)
    T = length(control_sequence)
    c₁ = control_sequence[1]
    P = transition_matrix(mc, c₁, parameters)
    state_sequence = Vector{Int}(undef, T)
    s₁ = rand(rng, Categorical(p0; check_args=check_args))
    state_sequence[1] = s₁
    @views for t in 1:(T - 1)
        cₜ = control_sequence[t]
        transition_matrix!(P, mc, cₜ, parameters)
        sₜ = state_sequence[t]
        sₜ₊₁ = rand(rng, Categorical(P[sₜ, :]; check_args=check_args))
        state_sequence[t + 1] = sₜ₊₁
    end
    return state_sequence
end

function Base.rand(
    mc::AbstractControlledMarkovChain,
    control_sequence::AbstractVector,
    parameters;
    check_args=false,
)
    return rand(GLOBAL_RNG, mc, control_sequence, parameters; check_args=check_args)
end
