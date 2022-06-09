
"""
    DiscreteMarkovChainPrior

Define a Dirichlet prior on the initial distribution and on the transition matrix of a [`DiscreteMarkovChain`](@ref).

# Fields
- `p0_α::Vector`: Dirichlet parameter for the initial distribution
- `P_α::Matrix`: Dirichlet parameters for the transition matrix
"""
struct DiscreteMarkovChainPrior{R1<:Real,R2<:Real} <: AbstractMarkovChainPrior
    p0_α::Vector{R1}
    P_α::Matrix{R2}
end

function flat_prior(mc::DiscreteMarkovChain{R1,R2}) where {R1<:Real,R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    P_α = ones(R2, S, S)
    return DiscreteMarkovChainPrior{R1,R2}(p0_α, P_α)
end

"""
    logdensityof(prior::DiscreteMarkovChainPrior, mc::DiscreteMarkovChain)

Compute the log-likelihood of the chain `mc` with respect to a `prior`.
"""
function DensityInterface.logdensityof(
    prior::DiscreteMarkovChainPrior, mc::AbstractDiscreteMarkovChain
)
    (; p0_α, P_α) = prior
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    l = logdensityof(Dirichlet(p0_α), p0)
    for s in 1:nb_states(mc)
        l += logdensityof(Dirichlet(view(P_α, s, :)), view(P, s, :))
    end
    return l
end

"""
    rand([rng,] prior::DiscreteMarkovChainPrior)

Sample a [`DiscreteMarkovChain`](@ref) from `prior`.
"""
function Base.rand(rng::AbstractRNG, prior::DiscreteMarkovChainPrior; check_args=false)
    (; p0_α, P_α) = prior
    p0 = rand(rng, Dirichlet(p0_α; check_args=check_args))
    P = reduce(
        vcat, rand(rng, Dirichlet(view(P_α, s, :); check_args=check_args)) for s in 1:S
    )
    return DiscreteMarkovChain(p0, P)
end

Base.rand(prior::DiscreteMarkovChainPrior; kwargs...) = rand(GLOBAL_RNG, prior; kwargs...)
