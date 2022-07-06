"""
    rand(rng, hmm::HMM, T[, control_sequence])

Sample a sequence of states of length `T` and the associated sequence of observations.
"""
function Base.rand(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    T::Integer,
    control_sequence::AbstractVector=Fill(nothing, T),
    ps=nothing,
    st=nothing;
    check_args=false,
)
    state_sequence = Vector{Int}(undef, T)
    p0 = initial_distribution(hmm)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        iₜ = state_sequence[t]
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, ps, st)
        iₜ₊₁ = rand(rng, Categorical(view(Pₜ, iₜ, :); check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end
    obs_sequence = [
        rand(
            rng, emission_distribution(hmm, state_sequence[t], control_sequence[t], ps, st)
        ) for t in 1:T
    ]
    return state_sequence, obs_sequence
end

function Base.rand(
    hmm::AbstractHMM,
    T::Integer,
    control_sequence::AbstractVector=Fill(nothing, T),
    check_args=false,
)
    return rand(GLOBAL_RNG, hmm, T, control_sequence; check_args=check_args)
end
