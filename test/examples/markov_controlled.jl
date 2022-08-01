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

struct NeuralMarkovChain{R,M} <: AbstractControlledMarkovChain
    p0::Vector{R}
    P_model::M
end

CHMMs.nb_states(nmc::NeuralMarkovChain) = length(nmc.p0)
CHMMs.initial_distribution(nmc::NeuralMarkovChain) = nmc.p0
CHMMs.transition_matrix(nmc::NeuralMarkovChain, u, ps, st) = first(nmc.P_model(u, ps, st))

##

U = 2
S = 3
T = 1000

p0 = rand_prob_vec(S)

P_model = Chain(
    Dense(U, 1), Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_row_stochastic
)
mc = NeuralMarkovChain(p0, P_model)

ps_true, st_true = Lux.setup(rng, P_model)
ps_init, st_init = Lux.setup(rng, P_model)
ps_init = ComponentVector(ps_init)

control_matrix = randn(U, T);
state_sequence = rand(mc, control_matrix, ps_true, st_true);

data = (mc, state_sequence, control_matrix, st_init)

function loss(ps, data)
    (mc, state_sequence, control_matrix, st) = data
    return -logdensityof(mc, state_sequence, control_matrix, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res = solve(prob, OptimizationOptimJL.LBFGS());
ps_est = res.u

logL_true = logdensityof(mc, state_sequence, control_matrix, ps_true, st_true)
logL_init = logdensityof(mc, state_sequence, control_matrix, ps_init, st_init)
logL_est = logdensityof(mc, state_sequence, control_matrix, ps_est, st_init)

@test logL_true > logL_init
@test logL_est > logL_true

P_true = transition_matrix(mc, ones(U, 1), ps_true, st_true)
P_init = transition_matrix(mc, ones(U, 1), ps_init, st_init)
P_est = transition_matrix(mc, ones(U, 1), ps_est, st_init)

err_init = mean(abs, P_true - P_init)
err_est = mean(abs, P_true - P_est)

@test err_est < err_init / 3
