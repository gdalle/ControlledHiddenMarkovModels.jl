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

## Normal HMM, single obs sequence

function add_suffstats(stats1::Distributions.NormalStats, stats2::Distributions.NormalStats)
    s = stats1.s + stats2.s
    s2 = stats1.s2 + stats2.s2
    tw = stats1.tw + stats2.tw
    m = s / tw
    stats = Distributions.NormalStats(s, m, s2, tw)
    return stats
end

function CHMMs.fit_mle_from_multiple_sequences(D::Type{<:Normal}, xs, ws)
    stats = reduce(add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws))
    return fit_mle(D, stats)
end

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

emissions_init = [Normal(μ_init[1], σ_init[1]), Normal(μ_init[2], σ_init[2])]
hmm_init = HMM(p0_init, P_init, emissions_init)

hmm_est, logL_evolution = baum_welch(obs_sequence, hmm_init; max_iterations=100, tol=1e-5);

p0_est = initial_distribution(hmm_est)
P_est = transition_matrix(hmm_est)
μ_est = [emission_distribution(hmm_est, s).μ for s in 1:2]
σ_est = [emission_distribution(hmm_est, s).σ for s in 1:2]

## Testing

P_error_init = mean(abs, P_init - P)
P_error = mean(abs, P_est - P)

μ_error_init = mean(abs, μ_init - μ)
μ_error = mean(abs, μ_est - μ)

σ_error_init = mean(abs, σ_init - σ)
σ_error = mean(abs, σ_est - σ)

l_init = logdensityof(hmm_init, obs_sequence)
l_est = logdensityof(hmm_est, obs_sequence)

@test P_error < P_error_init / 10
@test μ_error < μ_error_init / 10
@test σ_error < σ_error_init / 10
@test l_est > l_init
