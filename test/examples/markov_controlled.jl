using ComponentArrays
using ControlledHiddenMarkovModels
using Optimization
using ProgressMeter
using Random
using Statistics
using Test

using ForwardDiff: ForwardDiff
using OptimizationOptimisers: OptimizationOptimisers

rng = Random.default_rng()
Random.seed!(rng, 0)

## Struct

struct ControlledMarkovChain <: AbstractControlledMarkovChain
    S::Int
end

CHMMs.nb_states(mc::ControlledMarkovChain) = mc.S

function CHMMs.log_initial_distribution(mc::ControlledMarkovChain, parameters)
    return log.(ones(mc.S) ./ mc.S)
end

function CHMMs.log_transition_matrix!(
    logP::Matrix, ::ControlledMarkovChain, control::AbstractVector, parameters
)
    shift = sum(control) / length(control)
    logP .= parameters.logP .+ shift
    @views for s in axes(logP, 1)
        logP[s, :] .-= CHMMs.logsumexp_stream(logP[s, :])
    end
    return logP
end

function CHMMs.log_transition_matrix(
    mc::ControlledMarkovChain, control::AbstractVector, parameters
)
    logP = similar(parameters.logP)
    CHMMs.log_transition_matrix!(logP, mc, control, parameters)
    return logP
end

##

U = 2
S = 3
T = 1000

mc = ControlledMarkovChain(S)
parameters_true = ComponentVector(; logP=randn(rng, S, S))
parameters_init = ComponentVector(; logP=randn(rng, S, S))

control_sequence = [randn(rng, U) for t in 1:T];
state_sequence = rand(rng, mc, control_sequence, parameters_true);

data = (mc, state_sequence, control_sequence);

function loss(parameters, data)
    (mc, state_sequence, control_sequence) = data
    return -logdensityof(mc, state_sequence, control_sequence, parameters)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=1000);
parameters_est = res.u

logL_true = logdensityof(mc, state_sequence, control_sequence, parameters_true)
logL_init = logdensityof(mc, state_sequence, control_sequence, parameters_init)
logL_est = logdensityof(mc, state_sequence, control_sequence, parameters_est)

@test logL_est > logL_init

logP_true = log_transition_matrix(mc, ones(U), parameters_true)
logP_init = log_transition_matrix(mc, ones(U), parameters_init)
logP_est = log_transition_matrix(mc, ones(U), parameters_est)

err_init = mean(abs, logP_true - logP_init)
err_est = mean(abs, logP_true - logP_est)

@test err_est < err_init
