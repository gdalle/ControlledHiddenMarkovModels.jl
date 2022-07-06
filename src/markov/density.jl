"""
    logdensityof(mc::AbstractMarkovChain, state_sequence[, control_sequence])

Compute the log-likelihood of a sequence of integer states for the chain `mc`.
"""
function DensityInterface.logdensityof(
    mc::AbstractMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractVector=Fill(nothing, length(state_sequence)),
)
    T = length(state_sequence)
    i₁ = state_sequence[1]
    l = log(initial_distribution(mc)[i₁])
    for t in 1:(T - 1)
        iₜ, iₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        uₜ = control_sequence[t]
        l += log(transition_matrix(mc, uₜ)[iₜ, iₜ₊₁])
    end
    return l
end
