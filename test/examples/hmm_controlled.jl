using Combinatorics
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
using OptimizationOptimJL: OptimizationOptimJL
using OptimizationOptimisers: OptimizationOptimisers
using Zygote: Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 2
S = 3
T = 1000

p0 = rand_prob_vec(S)

## Neural Gaussian HMM

G = 3

P_μ_model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1 => S^2, softplus), ReshapeLayer((S, S)), make_row_stochastic),
        Chain(Dense(1 => S * G, identity), ReshapeLayer((G, S))),
    ),
)

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

function CHMMs.transition_matrix_and_emission_parameters(nhmm::NeuralGaussianHMM, u, ps, st)
    return nhmm.P_μ_model(u, ps, st)[1]
end

function CHMMs.emission_from_parameters(nhmm::NeuralGaussianHMM, μ::AbstractVector)
    return MvNormal(μ, I)
end

## Simulation

ps_true, st = Lux.setup(rng, P_μ_model);
ps_init, _ = Lux.setup(rng, P_μ_model);
ps_init = ComponentVector(ps_init);

nhmm = NeuralGaussianHMM(p0, P_μ_model);

control_matrix = randn(U, T);
state_sequence, obs_sequence = rand(nhmm, control_matrix, ps_true, st);

data = (nhmm, obs_sequence, control_matrix, st);

## Learning

function loss(ps, data)
    (nhmm, obs_sequence, control_matrix, st) = data
    return -logdensityof(nhmm, obs_sequence, control_matrix, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=1000);
ps_est = res.u;

## Testing

logL_true = logdensityof(nhmm, obs_sequence, control_matrix, ps_true, st)
logL_init = logdensityof(nhmm, obs_sequence, control_matrix, ps_init, st)
logL_est = logdensityof(nhmm, obs_sequence, control_matrix, ps_est, st)

@test logL_est > logL_init

single_control = randn(U, 1)
P_true = transition_matrix(nhmm, single_control, ps_true, st)
P_init = transition_matrix(nhmm, single_control, ps_init, st)
P_est = transition_matrix(nhmm, single_control, ps_est, st)

## Neural Poisson HMM

D = 5
V = 4

P_λ_p_model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1 => S^2, softplus), ReshapeLayer((S, S)), make_row_stochastic),
        Parallel(
            vcat,
            Chain(Dense(1 => S, softplus), ReshapeLayer((1, S))),
            Chain(
                Dense(1 => S * D * V, softplus),
                ReshapeLayer((V, D, S)),
                make_column_stochastic,
                ReshapeLayer((V * D, S)),
            ),
        ),
    ),
)

struct NeuralPoissonHMM{R,M} <: AbstractControlledHMM
    D::Int
    V::Int
    p0::Vector{R}
    P_λ_p_model::M
end

CHMMs.nb_states(nhmm::NeuralPoissonHMM) = length(nhmm.p0)
CHMMs.initial_distribution(nhmm::NeuralPoissonHMM) = nhmm.p0

function CHMMs.transition_matrix(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_p_model(u, ps, st)[1][1]
end

function CHMMs.emission_parameters(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_p_model(u, ps, st)[1][2]
end

function CHMMs.transition_matrix_and_emission_parameters(nhmm::NeuralPoissonHMM, u, ps, st)
    return nhmm.P_λ_p_model(u, ps, st)[1]
end

function CHMMs.emission_from_parameters(nhmm::NeuralPoissonHMM, θ::AbstractVector)
    λ = θ[1]
    p = @views reshape(θ[2:end], (nhmm.V, nhmm.D))
    return MarkedPoissonProcess(λ, p)
end

## Simulation

ps_true, st = Lux.setup(rng, P_λ_p_model);
ps_init, _ = Lux.setup(rng, P_λ_p_model);
ps_init = ComponentVector(ps_init);

nhmm = NeuralPoissonHMM(D, V, p0, P_λ_p_model);

control_matrix = rand(U, T);
state_sequence, obs_sequence = rand(nhmm, control_matrix, ps_true, st);

data = (nhmm, obs_sequence, control_matrix, st);

## Learning

function loss(ps, data)
    (nhmm, obs_sequence, control_matrix, st) = data
    return -logdensityof(nhmm, obs_sequence, control_matrix, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=100);
ps_est = res.u;

## Testing

logL_true = logdensityof(nhmm, obs_sequence, control_matrix, ps_true, st)
logL_init = logdensityof(nhmm, obs_sequence, control_matrix, ps_init, st)
logL_est = logdensityof(nhmm, obs_sequence, control_matrix, ps_est, st)

@test logL_est > logL_init
