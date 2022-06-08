function DensityInterface.logdensityof(mc::ContinuousMarkovChain, h::History{<:Integer})
    return error("not implemented")
end

function DensityInterface.logdensityof(
    prior::ContinuousMarkovChainPrior, mc::ContinuousMarkovChain
)
    l = logdensityof(Dirichlet(prior.p0α), mc.p0)
    Q = rates_matrix(mc)
    for i in 1:nb_states(mc), j in 1:nb_states(mc)
        if i != j
            gamma_prior_ij = Gamma(prior.Q_α[i, i], inv(prior.Q_β[i, j]))
            l += logdensityof(gamma_prior_ij, Q[i, j])
        end
    end
    return l
end
