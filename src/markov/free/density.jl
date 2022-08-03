function DensityInterface.logdensityof(
    mc::MarkovChain, state_sequence::AbstractVector{<:Integer}
)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    T = length(state_sequence)
    s₁ = state_sequence[1]
    logL = log(p0[s₁])
    for t in 1:(T - 1)
        sₜ, sₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        logL += log(P[sₜ, sₜ₊₁])
    end
    return logL
end
