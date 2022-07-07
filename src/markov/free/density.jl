function DensityInterface.logdensityof(
    mc::MarkovChain, state_sequence::AbstractVector{<:Integer}
)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    T = length(state_sequence)
    i₁ = state_sequence[1]
    logL = log(p0[i₁])
    for t in 1:(T - 1)
        iₜ, iₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        logL += log(P[iₜ, iₜ₊₁])
    end
    return logL
end
