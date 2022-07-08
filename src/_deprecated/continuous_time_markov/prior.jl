
"""
    ContinuousMarkovChainPrior

Define a Dirichlet prior on the initial distribution and a Gamma prior on the rates matrix of a [`ContinuousMarkovChain`](@ref).

# Fields
- `p0_α::Vector`: Dirichlet parameter for the initial distribution
- `Q_α::Matrix`: Gamma shape parameters for the rates matrix
- `Q_β::Vector`: Gamma rate parameters for the rates matrix
"""
struct ContinuousMarkovChainPrior{R1<:Real,R2<:Real,R3<:Real} <: AbstractMarkovChainPrior
    p0_α::Vector{R1}
    Q_α::Matrix{R2}
    Q_β::Vector{R3}
end

function flat_prior(mc::ContinuousMarkovChain{R1,R2}) where {R1<:Real,R2<:Real}
    S = nb_states(mc)
    p0_α = ones(R1, S)
    Q_α = ones(R2, S, S)
    Q_β = zeros(R2, S)
    return ContinuousMarkovChainPrior(p0_α, Q_α, Q_β)
end

"""
    rand([rng,] prior::ContinuousMarkovChainPrior)

Sample a [`ContinuousMarkovChain`](@ref) from `prior`.
"""
function Base.rand(
    rng::AbstractRNG, prior::ContinuousMarkovChainPrior{R1,R2,R3}; check_args=false
) where {R1<:Real,R2<:Real,R3<:Real}
    return error("Not implemented.")
end

Base.rand(prior::ContinuousMarkovChainPrior; kwargs...) = rand(GLOBAL_RNG, prior; kwargs...)

function DensityInterface.logdensityof(
    prior::ContinuousMarkovChainPrior, mc::ContinuousMarkovChain
)
    (; p0_α, Q_α, Q_β) = prior
    S = nb_states(mc)
    l = logdensityof(Dirichlet(p0_α), mc.p0)
    Q = rates_matrix(mc)
    for i in 1:S, j in 1:S
        if i != j
            gamma_prior_ij = Gamma(Q_α[i, i], inv(Q_β[i, j]))
            l += logdensityof(gamma_prior_ij, Q[i, j])
        end
    end
    return l
end
