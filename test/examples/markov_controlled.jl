using ComponentArrays
using ControlledHiddenMarkovModels
using Lux
using Optimization
using NNlib
using ProgressMeter
using Random
using Test

import ForwardDiff
import OptimizationFlux
import OptimizationOptimJL
import Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

make_stochastic(x) = x ./ sum(x; dims=2)

S = 4
T = 100

p0 = rand_prob_vec(S)

P_model = Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic)
mc = ControlledMarkovChain(p0, P_model)

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

loss(ps) = loss(ps, data)

@time ForwardDiff.gradient(loss, ps_init)
@time Zygote.gradient(loss, ps_init)

f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, ps_init, data)
@time res1 = solve(prob, OptimizationOptimJL.BFGS())
@time res2 = solve(prob, OptimizationFlux.Adam(), maxiters=1000)

ps_est1 = res1.u
ps_est2 = res2.u

ld_true = logdensityof(mc, state_sequence, control_sequence, ps_true, st_true)
ld_init = logdensityof(mc, state_sequence, control_sequence, ps_init, st_init)
ld_est1 = logdensityof(mc, state_sequence, control_sequence, ps_est1, st_init)
ld_est2 = logdensityof(mc, state_sequence, control_sequence, ps_est2, st_init)

@test ld_true > ld_init
@test ld_est1 > ld_true
@test ld_est2 > ld_true
