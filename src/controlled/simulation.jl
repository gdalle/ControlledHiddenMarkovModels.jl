"""
    rand(rng, hmm::AbstractControlledHMM, control_sequence, par)

Sample a trajectory from `hmm` with controls `control_sequence` and parameters `par`.

Returns a couple `(state_sequence, obs_sequence)`.
"""
function Base.rand(
    rng::AbstractRNG, hmm::AbstractControlledHMM, control_sequence, par; check_args=false
)
    T = length(control_sequence)

    p0 = initial_distribution(hmm, par)
    s₁ = rand(rng, Categorical(p0; check_args=check_args))
    state_sequence = Vector{typeof(s₁)}(undef, T)
    state_sequence[1] = s₁

    for t in 1:(T - 1)
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, par)
        sₜ = state_sequence[t]
        @views sₜ₊₁ = rand(rng, Categorical(Pₜ[sₜ, :]; check_args=check_args))
        state_sequence[t + 1] = sₜ₊₁
    end

    u₁ = control_sequence[1]
    θ₁ = emission_parameters(hmm, u₁, par)
    o₁ = rand(rng, emission_distribution(hmm, s₁, θ₁))
    obs_sequence = Vector{typeof(o₁)}(undef, T)
    obs_sequence[1] = o₁

    for t in 2:T
        uₜ = control_sequence[t]
        θₜ = emission_parameters(hmm, uₜ, par)
        sₜ = state_sequence[t]
        oₜ = rand(rng, emission_distribution(hmm, sₜ, θₜ))
        obs_sequence[t] = oₜ
    end

    return state_sequence, obs_sequence
end

function Base.rand(hmm::AbstractControlledHMM, control_sequence, par; check_args=false)
    return rand(GLOBAL_RNG, hmm, control_sequence, par; check_args=check_args)
end
