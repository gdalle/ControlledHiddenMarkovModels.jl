"""
    logdensityof(mc::AbstractDiscreteMarkovChain, state_sequence[, control_sequence])

Compute the log-likelihood of a sequence of integer states for the chain `mc`.
"""
function DensityInterface.logdensityof(
    mc::AbstractDiscreteMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractVector=Fill(nothing, length(state_sequence)),
    ps=nothing,
    st=nothing,
)
    T = length(state_sequence)
    l = log(initial_probability(mc, state_sequence[1]))
    for t in 1:(T - 1)
        iₜ, iₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        uₜ = control_sequence[t]
        l += log(transition_probability(mc, iₜ, iₜ₊₁, uₜ, ps, st))
    end
    return l
end
