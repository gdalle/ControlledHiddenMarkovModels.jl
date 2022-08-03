function Base.rand(rng::AbstractRNG, mc::MarkovChain, T::Integer; check_args=false)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    @views for t in 1:(T - 1)
        sₜ = state_sequence[t]
        sₜ₊₁ = rand(rng, Categorical(P[sₜ, :]; check_args=check_args))
        state_sequence[t + 1] = sₜ₊₁
    end
    return state_sequence
end

function Base.rand(mc::MarkovChain, T::Integer; check_args=false)
    return rand(GLOBAL_RNG, mc, T, control_matrix, args...; check_args=check_args)
end
