using ComponentArrays
using ControlledHiddenMarkovModels
using Lux
using Optimization
using NNlib
using ProgressMeter
using Random
using Statistics
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimJL: OptimizationOptimJL
using Zygote: Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

## Struct

struct ControlledMarkovChain <: AbstractControlledMarkovChain
    S::Int
end

CHMMs.nb_states(cmc::ControlledMarkovChain) = cmc.S
CHMMs.initial_distribution(cmc::ControlledMarkovChain) = ones(cmc.S) ./ cmc.S

function CHMMs.transition_matrix!(
    P::Matrix, ::ControlledMarkovChain, control::AbstractVector, params
)
    (; logP) = params
    scale = sum(abs, control)
    P .= scale .* exp.(logP)
    P ./= sum(P; dims=2)
    return P
end

function CHMMs.transition_matrix(
    cmc::ControlledMarkovChain, control::AbstractVector, params
)
    (; logP) = params
    scale = sum(abs, control)
    P = scale .* exp.(logP)
    P ./= sum(P; dims=2)
    return P
end

##

U = 5
S = 3
T = 1000

mc = ControlledMarkovChain(S)
params_true = ComponentVector(logP = randn(S, S),)
params_init = ComponentVector(logP = randn(S, S),)

control_sequence = [randn(U) for t in 1:T];
state_sequence = rand(mc, control_sequence, params_true);

data = (mc, state_sequence, control_sequence);

function loss(params, data)
    (mc, state_sequence, control_sequence) = data
    return -logdensityof(mc, state_sequence, control_sequence, params)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, params_init, data);
res = solve(prob, OptimizationOptimJL.LBFGS());
params_est = res.u

logL_true = logdensityof(mc, state_sequence, control_sequence, params_true)
logL_init = logdensityof(mc, state_sequence, control_sequence, params_init)
logL_est = logdensityof(mc, state_sequence, control_sequence, params_est)

@test logL_true > logL_init
@test logL_est > logL_true

P_true = transition_matrix(mc, ones(U), params_true)
P_init = transition_matrix(mc, ones(U), params_init)
P_est = transition_matrix(mc, ones(U), params_est)

err_init = mean(abs, P_true - P_init)
err_est = mean(abs, P_true - P_est)

@test err_est < err_init / 3
