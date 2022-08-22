# # Controlled Hidden Markov Model

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using ForwardDiff
using LinearAlgebra
using LogarithmicNumbers
using Optimization
using OptimizationOptimJL
using PointProcesses
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

## Controlled Normal HMM

struct ControlledNormalHMM <: AbstractControlledHMM end

CHMMs.nb_states(::ControlledNormalHMM, par) = length(par.logp0)

function CHMMs.initial_distribution(::ControlledNormalHMM, par)
    p0 = exp.(ULogarithmic, par.logp0)
    make_prob_vec!(p0)
    return p0
end

function CHMMs.transition_matrix!(P::AbstractMatrix, ::ControlledNormalHMM, control, par)
    P .= exp.(ULogarithmic, par.logP)
    make_trans_mat!(P)
    return P
end

function CHMMs.transition_matrix(::ControlledNormalHMM, control, par)
    P = exp.(ULogarithmic, par.logP)
    make_trans_mat!(P)
    return P
end

function CHMMs.emission_parameters!(
    θ::AbstractVector, hmm::ControlledNormalHMM, control, par
)
    mul!(θ.μ, par.μ_weights, control)
    mul!(θ.logσ, par.logσ_weights, control)
    return θ
end

function CHMMs.emission_parameters(hmm::ControlledNormalHMM, control, par)
    θ = ComponentVector(;
        μ=Vector{eltype(par.μ_weights)}(undef, nb_states(hmm, par)),
        logσ=Vector{eltype(par.logσ_weights)}(undef, nb_states(hmm, par)),
    )
    CHMMs.emission_parameters!(θ, hmm, control, par)
    return θ
end

function CHMMs.emission_distribution(::ControlledNormalHMM, s::Integer, θ)
    return Normal(θ.μ[s], exp(θ.logσ[s]))
end

## Simulation

U = 4
S = 3

par_true = ComponentVector(;
    logp0=log.(rand_prob_vec(rng, S)),
    logP=log.(rand_trans_mat(rng, S)),
    μ_weights=randn(rng, S, U),
    logσ_weights=randn(rng, S, U),
)

T = 100
control_sequence = [rand(rng, U) for t in 1:T];
obs_sequence = rand(rng, ControlledNormalHMM(), control_sequence, par_true)[2];

## Learning

par_init = ComponentVector(;
    logp0=log.(rand_prob_vec(rng, S)),
    logP=log.(rand_trans_mat(rng, S)),
    μ_weights=randn(rng, S, U),
    logσ_weights=randn(rng, S, U),
)

function loss(par)
    return -logdensityof(
        ControlledNormalHMM(), obs_sequence, control_sequence, par; safe=false
    )
end

ForwardDiff.gradient(p -> loss(p, data), par_init)

function Base.promote_rule(
    ::Type{R}, ::Type{ForwardDiff.Dual{T,V,N}}
) where {R<:(Union{Logarithmic{T},ULogarithmic{T}} where {T}),T,V,N}
    return ForwardDiff.Dual{T,promote_type(R,V),N}
end

ForwardDiff.gradient(p -> loss(p, data), par_init)
