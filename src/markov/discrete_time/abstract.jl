abstract type AbstractDiscreteMarkovChain <: AbstractMarkovChain end

"""
    transition_matrix(mc::AbstractDiscreteMarkovChain)

Return the transition matrix of `mc`.
"""
transition_matrix(mc::AbstractDiscreteMarkovChain) = error("not implemented")

"""
    transition_probability(mc::AbstractDiscreteMarkovChain, i, j)

Return the probability of `mc` transitioning from state `i` to state `j`.
"""
function transition_probability(mc::AbstractDiscreteMarkovChain, i::Integer, j::Integer)
    return error("not implemented")
end
