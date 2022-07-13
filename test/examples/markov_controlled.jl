using ComponentArrays
using ControlledHiddenMarkovModels
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

## Struct

struct NeuralMarkovChain{R,M} <: AbstractControlledMarkovChain
    p0::Vector{R}
    P_model::M
end

CHMMs.nb_states(nmc::NeuralMarkovChain) = length(nmc.p0)
CHMMs.initial_distribution(nmc::NeuralMarkovChain) = nmc.p0
CHMMs.transition_matrix(nmc::NeuralMarkovChain, u, ps, st) = first(nmc.P_model(u, ps, st))

##

make_stochastic(x) = x ./ sum(x; dims=2)

S = 3
T = 1000

p0 = rand_prob_vec(S)

P_model = Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic)
mc = NeuralMarkovChain(p0, P_model)

ps_true, st_true = Lux.setup(rng, P_model)
ps_init, st_init = Lux.setup(rng, P_model)
ps_init = ComponentVector(ps_init)

control_sequence = ones(1, T);
state_sequence = rand(mc, control_sequence, ps_true, st_true);

data = (mc, state_sequence, control_sequence, st_init)

function loss(ps, data)
    (mc, state_sequence, control_sequence, st) = data
    return -logdensityof(mc, state_sequence, control_sequence, ps, st)
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, ps_init, data);
res1 = solve(prob, OptimizationOptimJL.BFGS());
res2 = solve(prob, OptimizationOptimisers.Adam(); maxiters=1000);
ps_est1 = res1.u
ps_est2 = res2.u

logL_true = logdensityof(mc, state_sequence, control_sequence, ps_true, st_true)
logL_init = logdensityof(mc, state_sequence, control_sequence, ps_init, st_init)
logL_est1 = logdensityof(mc, state_sequence, control_sequence, ps_est1, st_init)
logL_est2 = logdensityof(mc, state_sequence, control_sequence, ps_est2, st_init)

@test logL_true > logL_init
@test logL_est1 > logL_true
@test logL_est2 > logL_true

P_true = transition_matrix(mc, ones(1, 1), ps_true, st_true)
P_init = transition_matrix(mc, ones(1, 1), ps_init, st_init)
P_est1 = transition_matrix(mc, ones(1, 1), ps_est1, st_init)
P_est2 = transition_matrix(mc, ones(1, 1), ps_est2, st_init)

err_init = mean(abs, P_true - P_init)
err_est1 = mean(abs, P_true - P_est1)
err_est2 = mean(abs, P_true - P_est2)

@test err_est1 < err_init / 3
@test err_est2 < err_init / 3
