abstract type AbstractDiscreteMarkovChain <: AbstractMarkovChain end

"""
    transition_matrix(mc::AbstractDiscreteMarkovChain[, args...])

Return the transition matrix of `mc`.
"""
transition_matrix(mc::AbstractDiscreteMarkovChain, args...) = error("not implemented")

"""
    transition_probability(mc::AbstractDiscreteMarkovChain, i, j[, args...])

Return the probability of `mc` transitioning from state `i` to state `j`.
"""
function transition_probability(
    mc::AbstractDiscreteMarkovChain, i::Integer, j::Integer, args...
)
    return error("not implemented")
end
