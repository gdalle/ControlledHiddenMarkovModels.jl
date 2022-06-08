"""
    nb_states(mc::DiscreteMarkovChain)

Return the number of states of `mc`.
"""
nb_states(mc::DiscreteMarkovChain) = length(mc.p0)

"""
    initial_distribution(mc::DiscreteMarkovChain)

Return the vector of initial state probabilities of `mc`.
"""
initial_distribution(mc::DiscreteMarkovChain) = mc.p0

"""
    transition_matrix(mc::DiscreteMarkovChain)

Return the matrix of transition probabilities of `mc`.
"""
transition_matrix(mc::DiscreteMarkovChain) = mc.P

"""
    stationary_distribution(mc::DiscreteMarkovChain)

Compute the equilibrium distribution of `mc` using its eigendecomposition.
"""
function stationary_distribution(mc::DiscreteMarkovChain)
    p_stat = real.(eigvecs(transition_matrix(mc)')[:, end])
    return p_stat / sum(p_stat)
end

"""
    zero_prior(mc::DiscreteMarkovChain)

Build a flat prior, for which MAP is equivalent to MLE.
"""
function flat_prior(mc::DiscreteMarkovChain{R1,R2}) where {R1<:Real, R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    P_α = ones(R2, S, S)
    return DiscreteMarkovChainPrior(p0_α, P_α)
end
