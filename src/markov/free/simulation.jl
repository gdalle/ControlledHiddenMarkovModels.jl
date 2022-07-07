function Base.rand(rng::AbstractRNG, mc::MarkovChain, T::Integer; check_args=false)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        iₜ = state_sequence[t]
        P_row = @view P[iₜ, :]
        iₜ₊₁ = rand(rng, Categorical(P_row; check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end
    return state_sequence
end

function Base.rand(mc::MarkovChain, T::Integer; check_args=false)
    return rand(GLOBAL_RNG, mc, T, control_sequence, args...; check_args=check_args)
end
