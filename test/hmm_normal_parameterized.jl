module HMMNormalParameterizedTest

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using ForwardDiff
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

p0_init = rand_prob_vec(rng, Float32, 2)
P_init = rand_trans_mat(rng, Float32, 2)
μ_init = [1.0f0, -1.0f0]
σ_init = [1.0f0, 1.0f0]

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

f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, par_init, obs_sequence)
res = solve(prob, OptimizationOptimJL.LBFGS())
par_est = res.u

hmm_init = HMM(
    initial_distribution(NormalHMM(), par_init),
    transition_matrix(NormalHMM(), par_init),
    [emission_distribution(NormalHMM(), s, par_init) for s in 1:2],
)

hmm_est = HMM(
    initial_distribution(NormalHMM(), par_est),
    transition_matrix(NormalHMM(), par_est),
    [emission_distribution(NormalHMM(), s, par_est) for s in 1:2],
)

## Testing

p0_est = initial_distribution(hmm_est)
P_est = transition_matrix(hmm_est)
μ_est = [emission_distribution(hmm_est, s).μ for s in 1:2]
σ_est = [emission_distribution(hmm_est, s).σ for s in 1:2]

P_error_init = mean(abs, P_init - P)
P_error = mean(abs, P_est - P)

μ_error_init = mean(abs, μ_init - μ)
μ_error = mean(abs, μ_est - μ)

σ_error_init = mean(abs, σ_init - σ)
σ_error = mean(abs, σ_est - σ)

l_init = logdensityof(hmm_init, obs_sequence)
l_est = logdensityof(hmm_est, obs_sequence)

@test P_error < P_error_init / 5
@test μ_error < μ_error_init / 5
@test σ_error < σ_error_init / 5
@test l_est > l_init

end
