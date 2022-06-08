"""
    logdensityof(mc::AbstractControlledDiscreteMarkovChain, state_sequence, control_sequence)

Compute the log-likelihood of a sequence of integer states for the chain `mc` given a sequence of controls
"""
function DensityInterface.logdensityof(
    mc::AbstractControlledDiscreteMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractVector,
)
    T = length(state_sequence)
    l = log(initial_probability(mc, first(state_sequence)))
    for t in 2:T
        i, j = state_sequence[t - 1], state_sequence[t]
        u = control_sequence[t - 1]
        l += log(transition_probability(mc, i, j, u))
    end
    return l
end
