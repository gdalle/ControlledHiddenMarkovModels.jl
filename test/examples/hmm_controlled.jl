using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using LinearAlgebra
using Lux
using Optimization
using NNlib
using ProgressMeter
using Random
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimisers: OptimizationOptimisers
using OptimizationOptimJL: OptimizationOptimJL
using Zygote: Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 2
S = 3
M = 4
T = 1000

make_stochastic(x) = x ./ sum(x; dims=2)

p0 = rand_prob_vec(S)

## Neural Gaussian HMM

struct NeuralGaussianHMM{R,M} <: AbstractControlledHMM
    p0::Vector{R}
    P_μ_model::M
end

CHMMs.nb_states(nhmm::NeuralGaussianHMM) = length(nhmm.p0)
CHMMs.initial_distribution(nhmm::NeuralGaussianHMM) = nhmm.p0

function CHMMs.transition_matrix(nhmm::NeuralGaussianHMM, u, ps, st)
    return nhmm.P_μ_model(u, ps, st)[1][1]
end

function CHMMs.emission_parameters(nhmm::NeuralGaussianHMM, u, ps, st)
    return nhmm.P_μ_model(u, ps, st)[1][2]
end

function CHMMs.emission_from_parameters(nhmm::NeuralGaussianHMM, μ::AbstractVector)
    return Normal(μ[1])
end

function CHMMs.transition_matrix_and_emission_parameters(nhmm::NeuralGaussianHMM, u, ps, st)
    return nhmm.P_μ_model(u, ps, st)[1]
end

## Architecture

P_μ_model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic),
        Chain(Dense(1, S), ReshapeLayer((1, S))),
    ),
)

## Simulation

ps_true, st_true = Lux.setup(rng, P_μ_model);
ps_init, st_init = Lux.setup(rng, P_μ_model);
ps_init = ComponentVector(ps_init);

nhmm = NeuralGaussianHMM(p0, P_μ_model);

control_sequence = randn(U, T);
state_sequence, obs_sequence = rand(nhmm, control_sequence, ps_true, st_true);

data = (nhmm, obs_sequence, control_sequence, st_init);

## Learning

function loss(ps, data)
    (nhmm, obs_sequence, control_sequence, st) = data
    return -logdensityof(nhmm, obs_sequence, control_sequence, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=1000);
ps_est = res.u;

## Testing

logL_true = logdensityof(nhmm, obs_sequence, control_sequence, ps_true, st_true)
logL_init = logdensityof(nhmm, obs_sequence, control_sequence, ps_init, st_init)
logL_est = logdensityof(nhmm, obs_sequence, control_sequence, ps_est, st_init)

@test logL_true > logL_init
@test logL_est > logL_true

## Neural Poisson HMM

struct NeuralPoissonHMM{R,M} <: AbstractControlledHMM
    p0::Vector{R}
    P_λ_model::M
end

CHMMs.nb_states(nhmm::NeuralPoissonHMM) = length(nhmm.p0)
CHMMs.initial_distribution(nhmm::NeuralPoissonHMM) = nhmm.p0

function CHMMs.transition_matrix(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_model(u, ps, st)[1][1]
end

function CHMMs.emission_parameters(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_model(u, ps, st)[1][2]
end

function CHMMs.emission_from_parameters(nhmm::NeuralPoissonHMM, λ::AbstractVector)
    return MultivariatePoissonProcess(λ)
end

function CHMMs.transition_matrix_and_emission_parameters(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_model(u, ps, st)[1]
end

## Architecture

P_λ_model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic),
        Chain(Dense(1, S * M, softplus), ReshapeLayer((M, S))),
    ),
)

## Simulation

ps_true, st_true = Lux.setup(rng, P_λ_model);
ps_init, st_init = Lux.setup(rng, P_λ_model);
ps_init = ComponentVector(ps_init);

nhmm = NeuralPoissonHMM(p0, P_λ_model);

control_sequence = rand(U, T);
state_sequence, obs_sequence = rand(nhmm, control_sequence, ps_true, st_true);

data = (nhmm, obs_sequence, control_sequence, st_init);

## Learning

function loss(ps, data)
    (nhmm, obs_sequence, control_sequence, st) = data
    return -logdensityof(nhmm, obs_sequence, control_sequence, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=1000);
ps_est = res.u;

## Testing

logL_true = logdensityof(nhmm, obs_sequence, control_sequence, ps_true, st_true)
logL_init = logdensityof(nhmm, obs_sequence, control_sequence, ps_init, st_init)
logL_est = logdensityof(nhmm, obs_sequence, control_sequence, ps_est, st_init)

@test logL_true > logL_init
@test logL_est > logL_true
