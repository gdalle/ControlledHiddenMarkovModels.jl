module ControlledTest

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using ForwardDiff
using LinearAlgebra
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

## Controlled Normal HMM

struct ControlledNormalHMM <: AbstractControlledHMM end

CHMMs.nb_states(::ControlledNormalHMM, par) = length(par.logp0)

function CHMMs.initial_distribution(::ControlledNormalHMM, par)
    return make_prob_vec(exp.(par.logp0))
end

function CHMMs.log_initial_distribution(::ControlledNormalHMM, par)
    return make_log_prob_vec(par.logp0)
end

function CHMMs.transition_matrix(::ControlledNormalHMM, control, par)
    return make_trans_mat(exp.(par.logP))
end

function CHMMs.log_transition_matrix(::ControlledNormalHMM, control, par)
    return make_log_trans_mat(par.logP)
end

function CHMMs.emission_parameters(::ControlledNormalHMM, control, par)
    θ = (μ=par.μ_weights * control, logσ=par.logσ_weights * control)
    return θ
end

function CHMMs.emission_distribution(::ControlledNormalHMM, s::Integer, θ)
    return Normal(θ.μ[s], exp(θ.logσ[s]))
end

## Simulation

U = 4
S = 3

par_true = ComponentVector(;
    logp0=log.(rand_prob_vec(rng, S)),
    logP=log.(rand_trans_mat(rng, S)),
    μ_weights=randn(rng, S, U),
    logσ_weights=randn(rng, S, U),
)

T = 100
K = 2

control_sequences = [[rand(rng, U) for t in 1:T] for k in 1:K]

obs_sequences = [
    rand(rng, ControlledNormalHMM(), control_sequence, par_true)[2] for
    control_sequence in control_sequences
]

## Learning

par_init = ComponentVector(;
    logp0=log.(rand_prob_vec(rng, S)),
    logP=log.(rand_trans_mat(rng, S)),
    μ_weights=randn(rng, S, U),
    logσ_weights=randn(rng, S, U),
)

data = (obs_sequences, control_sequences)

function loss(par, data; safe=true)
    (obs_sequences, control_sequences) = data
    return -sum(
        logdensityof(ControlledNormalHMM(), obs_sequence, control_sequence, par; safe=safe)
        for (obs_sequence, control_sequence) in zip(obs_sequences, control_sequences)
    )
end

l1 = loss(par_init, data; safe=false)
l2 = loss(par_init, data; safe=true)
@test l1 ≈ l2

f = OptimizationFunction(loss, Optimization.AutoZygote())
prob = OptimizationProblem(f, par_init, data)
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=100)
par_est = res.u

## Testing

@test loss(par_est, data) < loss(par_init, data)

end
