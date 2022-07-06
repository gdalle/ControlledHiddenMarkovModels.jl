using ControlledHiddenMarkovModels
using Flux
using ForwardDiff
using Optimization
using OptimizationOptimisers: Adam
using ProgressMeter
using Random
using Test
using Zygote

Random.seed!(63)

U = 3
S = 5
T = 100

p0 = rand_prob_vec(S)
P_model_true = Chain(Dense(U, 1), Dense(1, S^2), make_square, make_stochastic)
mc_true = ControlledMarkovChain(p0, P_model_true)

control_sequence = [rand(U) for t in 1:T];
state_sequence = rand(mc_true, T, control_sequence);

P_model_init = Chain(Dense(U, 1), Dense(1, S^2), make_square, make_stochastic)
mc_init = ControlledMarkovChain(p0, P_model_init)

@test logdensityof(mc_true, state_sequence, control_sequence) >
    logdensityof(mc_init, state_sequence, control_sequence)

u0, restructure = Flux.destructure(P_model_init)
data = (state_sequence, control_sequence)

function loss(u, data)
    (state_sequence, control_sequence) = data
    P_model = restructure(u)
    mc = ControlledMarkovChain(p0, P_model)
    l = logdensityof(mc, state_sequence, control_sequence)
    return -l
end

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, data)
@time res = solve(prob, Adam(), maxiters = 100);

P_model_est = restructure(res.u)
mc_est = ControlledMarkovChain(p0, P_model_est)

@test logdensityof(mc_true, state_sequence, control_sequence) <
    logdensityof(mc_est, state_sequence, control_sequence)
