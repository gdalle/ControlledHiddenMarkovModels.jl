module ParameterizedTest

using ComponentArrays
using ControlledHiddenMarkovModels
using ForwardDiff
using LogarithmicNumbers
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Random
using Statistics
using Test
using Zygote

rng = Random.default_rng()
Random.seed!(rng, 63)

## Simulation

T = 1000
K = 5

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
μ = [2.0, -3.0]
σ = [0.5, 0.7]
emissions = [CHMMs.MyNormal(μ[1], σ[1]), CHMMs.MyNormal(μ[2], σ[2])]
hmm = HMM(p0, P, emissions)

obs_sequences = [rand(rng, hmm, T)[2] for k in 1:K]

## Parameterized Normal HMM

struct NormalHMM <: AbstractHMM end

CHMMs.nb_states(::NormalHMM, par) = length(par.logp0)

function CHMMs.initial_distribution(::NormalHMM, par)
    return make_prob_vec(exp.(par.logp0))
end

function CHMMs.log_initial_distribution(::NormalHMM, par)
    return make_log_prob_vec(par.logp0)
end

function CHMMs.transition_matrix(::NormalHMM, par)
    return make_trans_mat(exp.(par.logP))
end

function CHMMs.log_transition_matrix(::NormalHMM, par)
    return make_log_trans_mat(par.logP)
end

function CHMMs.emission_distribution(::NormalHMM, s::Integer, par)
    return CHMMs.MyNormal(par.μ[s], exp(par.logσ[s]))
end

## Learning

p0_init = rand_prob_vec(rng, 2)
P_init = rand_trans_mat(rng, 2)
μ_init = [1.0, -1.0]
σ_init = [1.0, 1.0]

par_init = ComponentVector(;
    logp0=log.(copy(p0_init)),
    logP=log.(copy(P_init)),
    μ=copy(μ_init),
    logσ=log.(copy(σ_init)),
)

function loss(par, obs_sequences; safe=false)
    return -sum(
        logdensityof(NormalHMM(), obs_sequence, par; safe=safe) for
        obs_sequence in obs_sequences
    )
end

l1 = loss(par_init, obs_sequences; safe=false)
l2 = loss(par_init, obs_sequences; safe=true)
@test l1 ≈ l2

g1 = ForwardDiff.gradient(par -> loss(par, obs_sequences; safe=false), par_init)
g2 = ForwardDiff.gradient(par -> loss(par, obs_sequences; safe=true), par_init)
g3 = Zygote.gradient(par -> loss(par, obs_sequences; safe=false), par_init)[1]
g4 = Zygote.gradient(par -> loss(par, obs_sequences; safe=true), par_init)[1]

@test g1 ≈ g2
@test g1 ≈ g3
@test g1 ≈ g4

function callback(x, args...)
    @show x
    return false
end

f = OptimizationFunction(loss, Optimization.AutoZygote())
prob = OptimizationProblem(f, par_init, obs_sequences)
# res = solve(prob, OptimizationOptimJL.LBFGS())  # error
res = solve(prob, OptimizationOptimisers.Adam(1e-2); maxiters=100)
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

@test loss(par_est, obs_sequences) < loss(par_init, obs_sequences)
@test P_error < P_error_init / 2
@test μ_error < μ_error_init / 2

end
