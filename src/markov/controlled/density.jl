function DensityInterface.logdensityof(
    mc::AbstractControlledMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractVector,
    params,
)
    p0 = initial_distribution(mc)
    T = length(state_sequence)
    s₁ = state_sequence[1]
    logL = log(p0[s₁])
    c₁ = control_sequence[1]
    P = transition_matrix(mc, c₁, params)
    for t in 1:(T - 1)
        sₜ, sₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        logL += log(P[sₜ, sₜ₊₁])
        cₜ₊₁ = control_sequence[t + 1]
        transition_matrix!(P, mc, cₜ₊₁, params)
    end
    return logL
end
