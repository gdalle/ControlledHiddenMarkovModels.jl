using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using LinearAlgebra
using Lux
using NNlib
using Optimization
using ProgressMeter
using Random
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimisers: OptimizationOptimisers

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 4
S = 3
T = 100

## Neural Gaussian HMM

struct NeuralGaussianHMM{M,S} <: AbstractControlledHMM
    S::Int
    model::M
    st::S
end

CHMMs.nb_states(hmm::NeuralGaussianHMM) = hmm.S
CHMMs.initial_distribution(hmm::NeuralGaussianHMM) = ones(hmm.S) / hmm.S

function CHMMs.transition_matrix(hmm::NeuralGaussianHMM, control, parameters)
    return dropdims(hmm.model(control, parameters, hmm.st)[1][1]; dims=3)
end

function CHMMs.emission_parameters(hmm::NeuralGaussianHMM, control, parameters)
    return dropdims(hmm.model(control, parameters, hmm.st)[1][2]; dims=3)
end

function CHMMs.emission_from_parameters(hmm::NeuralGaussianHMM, θ::AbstractVector)
    return MvNormal(θ, I)
end

## Simulation

G = 2

model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1 => S^2, softplus), ReshapeLayer((S, S)), make_row_stochastic),
        Chain(Dense(1 => S * G, identity), ReshapeLayer((G, S))),
    ),
)

parameters_true, st = Lux.setup(rng, model);
parameters_init, _ = Lux.setup(rng, model);
parameters_init = ComponentVector(parameters_init);

hmm = NeuralGaussianHMM(S, model, st);

control_sequence = [randn(U, 1) for t in 1:T];
state_sequence, obs_sequence = rand(hmm, control_sequence, parameters_true);

## Learning

data = (hmm, obs_sequence, control_sequence);

function loss(parameters, data)
    (hmm, obs_sequence, control_sequence) = data
    return -logdensityof(hmm, obs_sequence, control_sequence, parameters)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=100);
parameters_est = res.u;

## Testing

logL_true = logdensityof(hmm, obs_sequence, control_sequence, parameters_true)
logL_init = logdensityof(hmm, obs_sequence, control_sequence, parameters_init)
logL_est = logdensityof(hmm, obs_sequence, control_sequence, parameters_est)

@test logL_est > logL_init

## Neural Poisson HMM

struct NeuralPoissonHMM <: AbstractControlledHMM
    S::Int
end

CHMMs.nb_states(hmm::NeuralPoissonHMM) = hmm.S
CHMMs.initial_distribution(hmm::NeuralPoissonHMM) = ones(hmm.S) ./ hmm.S

function CHMMs.transition_matrix!(P::Matrix, hmm::NeuralPoissonHMM, control, parameters)
    (; logP) = parameters
    shift = sum(control) / length(control)
    P .= exp.(logP .+ shift)
    P ./= sum(P; dims=2)
    return P
end

function CHMMs.transition_matrix(hmm::NeuralPoissonHMM, control::AbstractVector, parameters)
    (; logP) = parameters
    P = Matrix{float(eltype(logP))}(undef, size(logP)...)
    CHMMs.transition_matrix!(P, hmm, control, parameters)
    return P
end

function CHMMs.emission_parameters!(
    λ::AbstractMatrix, hmm::NeuralPoissonHMM, control, parameters
)
    (; logλ) = parameters
    shift = sum(control) / length(control)
    λ .= exp.(logλ .+ shift)
    return λ
end

function CHMMs.emission_parameters(hmm::NeuralPoissonHMM, control, parameters)
    (; logλ) = parameters
    λ = Matrix{float(eltype(logλ))}(undef, size(logλ)...)
    CHMMs.emission_parameters!(λ, hmm, control, parameters)
    return λ
end

function CHMMs.emission_from_parameters(hmm::NeuralPoissonHMM, λ::AbstractVector)
    return MultivariatePoissonProcess(λ)
end

## Simulation

M = 5

hmm = NeuralPoissonHMM(S);

parameters_true = ComponentVector(; logP=randn(S, S), logλ=randn(M, S))
parameters_init = ComponentVector(; logP=randn(S, S), logλ=randn(M, S))

control_sequence = [rand(U) for t in 1:T];

state_sequence, obs_sequence = rand(hmm, control_sequence, parameters_true);

## Learning

data = (hmm, obs_sequence, control_sequence);

function loss(parameters, data)
    (hmm, obs_sequence, control_sequence) = data
    return -logdensityof(hmm, obs_sequence, control_sequence, parameters)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=100);
parameters_est = res.u;

## Testing

logL_true = logdensityof(hmm, obs_sequence, control_sequence, parameters_true)
logL_init = logdensityof(hmm, obs_sequence, control_sequence, parameters_init)
logL_est = logdensityof(hmm, obs_sequence, control_sequence, parameters_est)

@test logL_est > logL_init
