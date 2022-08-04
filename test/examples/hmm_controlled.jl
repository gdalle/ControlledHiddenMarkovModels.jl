## Imports

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using LinearAlgebra
using Lux
using Optimization
using ProgressMeter
using Statistics
using Random
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimisers: OptimizationOptimisers
using NNlib: NNlib

## Setup

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 50
S = 5
T = 100
C = 40
V = 3

function loss(parameters, data)
    (hmm, obs_sequence, control_sequence) = data
    return -logdensityof(hmm, obs_sequence, control_sequence, parameters; log=true)
end

function callback(x, l)
    @info "Iteration" l
    return false
end

## Abstract Poisson HMM

abstract type ControlledPoissonHMM <: AbstractControlledHMM end

CHMMs.nb_states(hmm::ControlledPoissonHMM) = hmm.S
CHMMs.initial_distribution(hmm::ControlledPoissonHMM) = ones(hmm.S) / hmm.S

## Fast Poisson HMM

struct FastPoissonHMM <: ControlledPoissonHMM
    S::Int
end

function CHMMs.transition_matrix!(P::Matrix, hmm::FastPoissonHMM, control, parameters)
    (; logP) = parameters
    shift = mean(control)
    P .= logP .+ shift
    @views for s in axes(P, 1)
        P[s, :] .-= CHMMs.logsumexp(P[s, :])
    end
    P .= exp.(P)
    return P
end

function CHMMs.transition_matrix(hmm::FastPoissonHMM, control::AbstractVector, parameters)
    (; logP) = parameters
    P = Matrix{float(eltype(logP))}(undef, size(logP)...)
    CHMMs.transition_matrix!(P, hmm, control, parameters)
    return P
end

function CHMMs.emission_parameters!(
    θ::AbstractVector, hmm::FastPoissonHMM, control, parameters
)
    (; logλ, logp) = parameters.θ
    shift = mean(control)
    θ.logλ .= logλ .+ shift
    θ.logp .= logp .+ shift
    @views for s in axes(θ.logp, 3), c in axes(θ.logp, 2)
        θ.logp[:, c, s] .-= CHMMs.logsumexp(θ.logp[:, c, s])
    end
    return θ
end

function CHMMs.emission_parameters(hmm::FastPoissonHMM, control, parameters)
    θ = similar(parameters.θ)
    CHMMs.emission_parameters!(θ, hmm, control, parameters)
    return θ
end

function CHMMs.emission_from_parameters(hmm::FastPoissonHMM, θ::AbstractVector, s::Integer)
    logλ = θ.logλ[s]
    logp = @view θ.logp[:, :, s]
    return DelimitedPoissonProcess(LogMarkedPoissonProcess(logλ, logp), 0., 1.)
end

## Simulation

hmm = FastPoissonHMM(S);

parameters_true = ComponentVector(;
    logP=randn(S, S), θ=ComponentVector(; logλ=randn(S), logp=randn(V, C, S))
);
parameters_init = ComponentVector(;
    logP=randn(S, S), θ=ComponentVector(; logλ=randn(S), logp=randn(V, C, S))
);

control_sequence = [rand(U) for t in 1:T];
state_sequence, obs_sequence = rand(hmm, control_sequence, parameters_true);

## Learning

data = (hmm, obs_sequence, control_sequence);
f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=10);
# @time solve(prob, OptimizationOptimisers.Adam(); maxiters=10);
# @profview solve(prob, OptimizationOptimisers.Adam(); maxiters=10);

parameters_est = res.u;

## Testing

logL_true = logdensityof(hmm, obs_sequence, control_sequence, parameters_true)
logL_init = logdensityof(hmm, obs_sequence, control_sequence, parameters_init)
logL_est = logdensityof(hmm, obs_sequence, control_sequence, parameters_est)

@test logL_est > logL_init
