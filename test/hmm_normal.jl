module HMMNormalTest

using ControlledHiddenMarkovModels
using Distributions
using LogarithmicNumbers
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

## Simulation

T = 1000
K = 5

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]

μ = [2.0, -3.0]
σ = [0.5, 0.7]
emissions = [Normal(μ[1], σ[1]), Normal(μ[2], σ[2])]

hmm = HMM(p0, P, emissions)

obs_sequences = [rand(rng, hmm, T)[2] for k in 1:K]

## Learning

p0_init = rand_prob_vec(rng, Float32, 2)
P_init = rand_trans_mat(rng, Float32, 2)

p0_init_log = LogFloat32.(p0_init)
P_init_log = LogFloat32.(P_init)

μ_init = Float64.([1.0f0, -1.0f0])
σ_init = Float64.([1.0f0, 1.0f0])
emissions_init = [Normal(μ_init[1], σ_init[1]), Normal(μ_init[2], σ_init[2])]

hmm_inits = (
    HMM(p0_init, P_init, emissions_init), HMM(p0_init_log, P_init_log, emissions_init)
)

for hmm_init in hmm_inits
    hmm_ests = (
        baum_welch_nolog(obs_sequences, hmm_init)[1],
        baum_welch_log(obs_sequences, hmm_init)[1],
        baum_welch_doublelog(obs_sequences, hmm_init)[1],
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

        l_init = sum(logdensityof(hmm_init, obs_sequence) for obs_sequence in obs_sequences)
        l_est = sum(logdensityof(hmm_est, obs_sequence) for obs_sequence in obs_sequences)

        @test P_error < P_error_init / 5
        @test μ_error < μ_error_init / 5
        @test σ_error < σ_error_init / 5
        @test l_est > l_init
    end
end

end
