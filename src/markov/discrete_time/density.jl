"""
    logdensityof(mc::AbstractDiscreteMarkovChain, state_sequence)

Compute the log-likelihood of a sequence of integer states for the chain `mc`.
"""
function DensityInterface.logdensityof(
    mc::AbstractDiscreteMarkovChain, state_sequence::AbstractVector{<:Integer}
)
    T = length(state_sequence)
    l = log(initial_probability(mc, first(state_sequence)))
    for t in 2:T
        i, j = state_sequence[t - 1], state_sequence[t]
        l += log(transition_probability(mc, i, j))
    end
    return l
end
