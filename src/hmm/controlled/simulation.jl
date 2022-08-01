function Base.rand(
    rng::AbstractRNG,
    hmm::AbstractControlledHMM,
    control_matrix::AbstractMatrix,
    ps,
    st;
    check_args=false,
)
    T = size(control_matrix, 2)
    p0 = initial_distribution(hmm)
    P_all = transition_matrix(hmm, control_matrix, ps, st)
    state_sequence = Vector{Int}(undef, T)
    state_sequence[1] = rand(rng, Categorical(p0; check_args=check_args))
    for t in 1:(T - 1)
        iₜ = state_sequence[t]
        Pₜ_row = @view P_all[iₜ, :, t]
        iₜ₊₁ = rand(rng, Categorical(Pₜ_row; check_args=check_args))
        state_sequence[t + 1] = iₜ₊₁
    end

    θ_all = emission_parameters(hmm, control_matrix, ps, st)
    obs_sequence = @views [
        rand(rng, emission_from_parameters(hmm, θ_all[:, state_sequence[t], t])) for
        t in 1:T
    ]

    return state_sequence, obs_sequence
end

function Base.rand(
    hmm::AbstractControlledHMM, control_matrix::AbstractMatrix, ps, st; check_args=false
)
    return rand(GLOBAL_RNG, hmm, control_matrix, ps, st; check_args=check_args)
end
