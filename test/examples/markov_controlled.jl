using AbstractDifferentiation
using ControlledHiddenMarkovModels
using ForwardDiff
using Lux
using NNlib
using Optimisers
using ProgressMeter
using Random
using Test
using Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

backend = AD.ZygoteBackend()

make_stochastic(x) = x ./ sum(x; dims=2)

S = 3
T = 1000

p0 = rand_prob_vec(S)

P_model_true = Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic)
ps_true, st_true = Lux.setup(rng, P_model_true)
mc_true = ControlledMarkovChain(p0, P_model_true)

P_model = Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic)
ps, st = Lux.setup(rng, P_model)
mc = ControlledMarkovChain(p0, P_model)

control_sequence = ones(1, T);
state_sequence = rand(mc_true, control_sequence, ps_true, st_true);

@test logdensityof(mc_true, state_sequence, control_sequence, ps_true, st_true) >
    logdensityof(mc, state_sequence, control_sequence, ps, st)

data = (mc, state_sequence, control_sequence, st)

function loss(ps, data)
    (mc, state_sequence, control_sequence, st) = data
    return -logdensityof(mc, state_sequence, control_sequence, ps, st)
end

opt = Adam()
opt_st = Optimisers.setup(opt, ps)

@showprogress for i in 1:500
    global ps, st, data, opt_st
    gs = Zygote.gradient(loss, ps, data)[1]
    opt_st, ps = Optimisers.update(opt_st, ps, gs)
end

@test isapprox(
    transition_matrix(mc, ones(1, 1), ps, st),
    transition_matrix(mc_true, ones(1, 1), ps_true, st_true);
    atol=1e-1
)
