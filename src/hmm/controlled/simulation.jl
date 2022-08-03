function Base.rand(
    rng::AbstractRNG,
    hmm::AbstractControlledHMM,
    control_sequence::AbstractVector,
    parameters;
    check_args=false,
)
    T = length(control_sequence)
    p0 = initial_distribution(hmm)

    c₁ = control_sequence[1]
    P = transition_matrix(hmm, c₁, parameters)
    θ = emission_parameters(hmm, c₁, parameters)

    s₁ = rand(rng, Categorical(p0; check_args=check_args))
    state_sequence = Vector{typeof(s₁)}(undef, T)
    state_sequence[1] = s₁
    @views for t in 1:(T - 1)
        cₜ = control_sequence[t]
        transition_matrix!(P, hmm, cₜ, parameters)
        sₜ = state_sequence[t]
        sₜ₊₁ = rand(rng, Categorical(P[sₜ, :]; check_args=check_args))
        state_sequence[t + 1] = sₜ₊₁
    end

    o₁ = @views rand(rng, emission_from_parameters(hmm, θ[:, s₁]))
    obs_sequence = Vector{typeof(o₁)}(undef, T)
    obs_sequence[1] = o₁
    @views for t in 2:T
        cₜ = control_sequence[t]
        emission_parameters!(θ, hmm, cₜ, parameters)
        sₜ = state_sequence[t]
        oₜ = rand(rng, emission_from_parameters(hmm, θ[:, sₜ]))
        obs_sequence[t] = oₜ
    end

    return state_sequence, obs_sequence
end

function Base.rand(
    hmm::AbstractControlledHMM,
    control_sequence::AbstractVector,
    parameters;
    check_args=false,
)
    return rand(GLOBAL_RNG, hmm, control_sequence, parameters; check_args=check_args)
end
