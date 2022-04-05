function DensityInterface.logdensityof(mc::DiscreteMarkovChain, x::AbstractVector)
    T = length(x)
    l = log(mc.π0[x[1]])
    for t in 2:T
        l += log(mc.P[x[t - 1], x[t]])
    end
    return l
end

function DensityInterface.logdensityof(
    prior::DiscreteMarkovChainPrior, mc::DiscreteMarkovChain
)
    l = logdensityof(Dirichlet(prior.π0_α), Categorical(mc.π0))
    for s in 1:nb_states(mc)
        l += logdensityof(Dirichlet(@view prior.P_α[s, :]), Categorical(@view mc.P[s, :]))
    end
    return l
end
