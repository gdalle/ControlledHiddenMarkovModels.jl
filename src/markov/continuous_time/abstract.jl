abstract type AbstractContinuousMarkovChain <: AbstractMarkovChain end

"""
    intensity_matrix(mc::AbstractContinuousMarkovChain)

Return the intensity matrix of `mc`.
"""
intensity_matrix(mc::AbstractContinuousMarkovChain) = error("not implemented")

"""
    intensity_matrix(mc::AbstractContinuousMarkovChain)

Return the intensity of the transition from `i` to `j` for `mc`.
"""
function intensity_value(mc::AbstractContinuousMarkovChain, i::Integer, j::Integer)
    return intensity_matrix(mc)[i, j]
end

"""
    intensity_negdiag(mc::AbstractContinuousMarkovChain)

Return the negative diagonal of the transition intensity matrix of `mc`.

This gives the departure intensities of the chain from each state.
"""
function intensity_negdiag(mc::AbstractContinuousMarkovChain)
    return -diag(intensity_matrix(mc))
end

"""
    embedded_transition_matrix(mc::AbstractContinuousMarkovChain)

Compute the transition matrix of the embedded discrete-time Markov chain.
"""
function embedded_transition_matrix(mc::AbstractContinuousMarkovChain)
    P = intensity_matrix(mc) ./ intensity_negdiag(mc)
    P[diagind(P)] .= zero(eltype(P))
    return P
end
