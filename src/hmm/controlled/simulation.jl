function Base.rand(
    rng::AbstractRNG,
    hmm::AbstractControlledHMM,
    control_sequence::AbstractMatrix,
    args...;
    check_args=false,
)
    T = length(control_sequence)
    p0 = initial_distribution(hmm)
    P_all = transition_matrix(hmm, control_sequence, args...)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        Pₜ = view(P_all, :, :, t)
        iₜ = state_sequence[t]
        iₜ₊₁ = rand(rng, Categorical(view(Pₜ, iₜ, :); check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end

    θ_all = emission_parameters(hmm, control_sequence, args...)
    obs_sequence = [
        rand(rng, emission_from_parameters(hmm, view(θ_all, :, state_sequence[t], t))) for
        t in 1:T
    ]

    return state_sequence, obs_sequence
end

function Base.rand(
    hmm::AbstractControlledHMM, control_sequence::AbstractMatrix, args...; check_args=false
)
    return rand(GLOBAL_RNG, hmm, control_sequence, args...; check_args=check_args)
end
