module HMMNormalTest

using ControlledHiddenMarkovModels
using Distributions
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

## Simulation

T = 1000

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
emissions_init = [Normal(μ_init[1], σ_init[1]), Normal(μ_init[2], σ_init[2])]

hmm_init = HMM(p0_init, P_init, emissions_init)

hmm_ests = (
    baum_welch(obs_sequence, hmm_init;)[1], baum_welch_log(obs_sequence, hmm_init;)[1]
)

for hmm_est in hmm_ests
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

    @test typeof(hmm_est) == typeof(hmm_init)
    @test P_error < P_error_init / 5
    @test μ_error < μ_error_init / 5
    @test σ_error < σ_error_init / 5
    @test l_est > l_init
end

end
