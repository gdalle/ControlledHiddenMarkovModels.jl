using ComponentArrays
using ControlledHiddenMarkovModels
using Optimization
using ProgressMeter
using Random
using Statistics
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimJL: OptimizationOptimJL

rng = Random.default_rng()
Random.seed!(rng, 0)

## Struct

struct ControlledMarkovChain <: AbstractControlledMarkovChain
    S::Int
end

CHMMs.nb_states(mc::ControlledMarkovChain) = mc.S
CHMMs.initial_distribution(mc::ControlledMarkovChain) = ones(mc.S) ./ mc.S

function CHMMs.transition_matrix!(
    P::Matrix, ::ControlledMarkovChain, control::AbstractVector, parameters
)
    (; logP) = parameters
    shift = sum(control) / length(control)
    P .= exp.(logP .+ shift)
    P ./= sum(P; dims=2)
    return P
end

function CHMMs.transition_matrix(
    mc::ControlledMarkovChain, control::AbstractVector, parameters
)
    (; logP) = parameters
    S = nb_states(mc)
    P = Matrix{float(eltype(logP))}(undef, S, S)
    CHMMs.transition_matrix!(P, mc, control, parameters)
    return P
end

##

U = 2
S = 3
T = 1000

mc = ControlledMarkovChain(S)
parameters_true = ComponentVector(; logP=randn(S, S))
parameters_init = ComponentVector(; logP=randn(S, S))

control_sequence = [randn(U) for t in 1:T];
state_sequence = rand(mc, control_sequence, parameters_true);

data = (mc, state_sequence, control_sequence);

function loss(parameters, data)
    (mc, state_sequence, control_sequence) = data
    return -logdensityof(mc, state_sequence, control_sequence, parameters)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimJL.LBFGS());
parameters_est = res.u

logL_true = logdensityof(mc, state_sequence, control_sequence, parameters_true)
logL_init = logdensityof(mc, state_sequence, control_sequence, parameters_init)
logL_est = logdensityof(mc, state_sequence, control_sequence, parameters_est)

@test logL_true > logL_init
@test logL_est > logL_true

P_true = transition_matrix(mc, ones(U), parameters_true)
P_init = transition_matrix(mc, ones(U), parameters_init)
P_est = transition_matrix(mc, ones(U), parameters_est)

err_init = mean(abs, P_true - P_init)
err_est = mean(abs, P_true - P_est)

@test err_est < err_init / 3
