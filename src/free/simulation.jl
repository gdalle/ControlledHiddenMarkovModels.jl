function Base.rand(
    rng::AbstractRNG, hmm::AbstractHMM, T::Integer, par=nothing; check_args=false
)
    S = nb_states(hmm, par)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    state_sequence = Vector{Int}(undef, T)
    s = rand(rng, Categorical(p0; check_args=check_args))
    state_sequence[1] = s
    @views for t in 2:T
        s = rand(rng, Categorical(P[s, :]; check_args=check_args))
        state_sequence[t] = s
    end
    obs_sequence = [rand(rng, emissions[state_sequence[t]]) for t in 1:T]
    return state_sequence, obs_sequence
end

function Base.rand(hmm::AbstractHMM, T::Integer, par=nothing; check_args=false)
    return rand(GLOBAL_RNG, hmm, T, par; check_args=check_args)
end
