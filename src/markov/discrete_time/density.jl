"""
    logdensityof(mc::DiscreteMarkovChain, x::AbstractVector)

Compute the log-likelihood of the sequence `x` of states for the chain `mc`.
"""
function DensityInterface.logdensityof(mc::DiscreteMarkovChain, x::AbstractVector)
    T = length(x)
    l = log(mc.p0[x[1]])
    for t in 2:T
        l += log(mc.P[x[t - 1], x[t]])
    end
    return l
end

"""
    logdensityof(prior::DiscreteMarkovChainPrior, mc::DiscreteMarkovChain)

Compute the log-likelihood of the chain `mc` with respect to a `prior`.
"""
function DensityInterface.logdensityof(
    prior::DiscreteMarkovChainPrior, mc::DiscreteMarkovChain
)
    l = logdensityof(Dirichlet(prior.p0_α), mc.p0)
    for s in 1:nb_states(mc)
        l += logdensityof(Dirichlet(@view prior.P_α[s, :]), @view mc.P[s, :])
    end
    return l
end
