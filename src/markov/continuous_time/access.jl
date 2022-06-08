"""
    nb_states(mc::ContinuousMarkovChain)

Return the number of states of `mc`.
"""
nb_states(mc::ContinuousMarkovChain) = length(mc.p0)

"""
    initial_distribution(mc::ContinuousMarkovChain)

Return the vector of initial state probabilities of `mc`.
"""
initial_distribution(mc::ContinuousMarkovChain) = mc.p0

"""
    rates_matrix(mc::ContinuousMarkovChain)

Return the matrix of transition rates of `mc`.
"""
rates_matrix(mc::ContinuousMarkovChain) = mc.Q

rates_negdiag(mc::ContinuousMarkovChain) = -diag(rates_matrix(mc))

function embedded_transition_matrix(mc::ContinuousMarkovChain)
    P = rates_matrix(mc) ./ rates_negdiag(mc)
    P[diagind(P)] .= zero(eltype(P))
    return P
end

function stationary_distribution(mc::ContinuousMarkovChain)
    p = real.(eigvecs(rates_matrix(mc)')[:, end])
    return p / sum(p)
end

"""
    zero_prior(mc::ContinuousMarkovChain)

Build a flat prior, for which MAP is equivalent to MLE.
"""
function flat_prior(mc::ContinuousMarkovChain{R1,R2}) where {R1<:Real,R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    Q_α = ones(R2, S, S)
    Q_β = zeros(R2, S)
    return ContinuousMarkovChainPrior(p0_α, Q_α, Q_β)
end
