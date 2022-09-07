# # Hidden Markov Model

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using ForwardDiff
using LogarithmicNumbers
using Optimization
using OptimizationOptimJL
using PointProcesses
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

## Simulation

T = 2000

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
μ = [2.0, -3.0]
σ = [0.5, 0.7]
emissions = [Normal(μ[1], σ[1]), Normal(μ[2], σ[2])]
hmm = HMM(p0, P, emissions)

obs_sequence = rand(rng, hmm, T)[2]

## Learning

p0_init = rand_prob_vec(rng, LogFloat32, 2)
P_init = rand_trans_mat(rng, LogFloat32, 2)
μ_init = [1.0, -1.0]
σ_init = ones(2)

## Parameterized Normal HMM

struct NormalHMM <: AbstractHMM end

CHMMs.nb_states(::NormalHMM, par) = length(par.logp0)

function CHMMs.initial_distribution(::NormalHMM, par)
    p0 = exp.(par.logp0)
    make_prob_vec!(p0)
    return p0
end

function CHMMs.log_initial_distribution(::NormalHMM, par)
    logp0 = copy(par.logp0)
    make_log_prob_vec!(logp0)
    return logp0
end

function CHMMs.transition_matrix(::NormalHMM, par)
    P = exp.(par.logP)
    make_trans_mat!(P)
    return P
end

function CHMMs.log_transition_matrix(::NormalHMM, par)
    logP = copy(par.logP)
    make_log_trans_mat!(logP)
    return logP
end

function CHMMs.emission_distribution(::NormalHMM, s::Integer, par)
    return Normal(par.μ[s], exp(par.logσ[s]))
end

## Learning

par_init = ComponentVector(;
    logp0=log.(copy(p0_init)),
    logP=log.(copy(P_init)),
    μ=copy(μ_init),
    logσ=log.(copy(σ_init)),
)

function loss(par, obs_sequence; safe=true)
    return -logdensityof(NormalHMM(), obs_sequence, par; safe=safe)
end

@test loss(par_init, obs_sequence; safe=true) ≈ loss(par_init, obs_sequence; safe=false)

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, par_init, obs_sequence);
res = solve(prob, OptimizationOptimJL.LBFGS());
par_est = res.u;

hmm_est2 = HMM(
    initial_distribution(NormalHMM(), par_est),
    transition_matrix(NormalHMM(), par_est),
    [emission_distribution(NormalHMM(), s, par_est) for s in 1:2],
)

## Testing

p0_est2 = initial_distribution(hmm_est2)
P_est2 = transition_matrix(hmm_est2)
μ_est2 = [emission_distribution(hmm_est2, s).μ for s in 1:2]
σ_est2 = [emission_distribution(hmm_est2, s).σ for s in 1:2]

P_error2 = mean(abs, P_est2 - P)
μ_error2 = mean(abs, μ_est2 - μ)
σ_error2 = mean(abs, σ_est2 - σ)
l_est2 = logdensityof(hmm_est2, obs_sequence)

@test P_error2 < P_error_init / 10
@test μ_error2 < μ_error_init / 10
@test σ_error2 < σ_error_init / 10
@test l_est2 > l_init
