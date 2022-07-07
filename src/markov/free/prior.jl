
"""
    MarkovChainPrior

Define a Dirichlet prior on the initial distribution and on the transition matrix of a [`MarkovChain`](@ref).

# Fields
- `p0_α::Vector`: Dirichlet parameter for the initial distribution
- `P_α::Matrix`: Dirichlet parameters for the transition matrix
"""
struct MarkovChainPrior{R1<:Real,R2<:Real}
    p0_α::Vector{R1}
    P_α::Matrix{R2}
end

@inline DensityInterface.DensityKind(::MarkovChainPrior) = HasDensity()

"""
    flat_prior(mc::MarkovChain)

Build a flat prior for `mc`, under which MAP is equivalent to MLE.
"""
function flat_prior(mc::MarkovChain{R1,R2}) where {R1<:Real,R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    P_α = ones(R2, S, S)
    return MarkovChainPrior{R1,R2}(p0_α, P_α)
end

"""
    logdensityof(prior::MarkovChainPrior, mc::MarkovChain)

Compute the log-likelihood of the chain `mc` with respect to a `prior`.
"""
function DensityInterface.logdensityof(prior::MarkovChainPrior, mc::MarkovChain)
    (; p0_α, P_α) = prior
    S = length(p0_α)
    p0 = initial_distribution(mc)
    P = transition_matrix(mc)
    l = logdensityof(Dirichlet(p0_α), p0)
    for i in 1:S
        l += logdensityof(Dirichlet(view(P_α, i, :)), view(P, i, :))
    end
    return l
end

"""
    rand([rng,] prior::MarkovChainPrior)

Sample a [`MarkovChain`](@ref) from `prior`.
"""
function Base.rand(rng::AbstractRNG, prior::MarkovChainPrior; check_args=false)
    (; p0_α, P_α) = prior
    S = length(p0_α)
    p0 = rand(rng, Dirichlet(p0_α; check_args=check_args))
    P = reduce(
        vcat, rand(rng, Dirichlet(view(P_α, i, :); check_args=check_args)) for i in 1:S
    )
    return MarkovChain(p0, P)
end

Base.rand(prior::MarkovChainPrior; kwargs...) = rand(GLOBAL_RNG, prior; kwargs...)
