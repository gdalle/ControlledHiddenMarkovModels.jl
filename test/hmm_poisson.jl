module HMMPoissonTest

using ControlledHiddenMarkovModels
using Distributions
using LogarithmicNumbers
using PointProcesses
using Random
using Statistics
using Test

rng = Random.default_rng()
Random.seed!(rng, 63)

function CHMMs.fit_mle_from_multiple_sequences(
    D::Type{<:MultivariatePoissonProcess}, xs, ws
)
    stats = reduce(
        PointProcesses.add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws)
    )
    return fit_mle(D, stats)
end

## Simulation

T = 100
K = 5

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]

λ = [1.0 3.0; 2.0 2.0; 3.0 1.0]
emissions = [
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 1]), 0.0, 1.0),
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 2]), 0.0, 1.0),
]

hmm = HMM(p0, P, emissions)

obs_sequences = [rand(rng, hmm, T)[2] for k in 1:K]

## Learning

p0_init = rand_prob_vec(rng, Float32, 2)
P_init = rand_trans_mat(rng, Float32, 2)

p0_init_log = LogFloat32.(p0_init)
P_init_log = LogFloat32.(P_init)

λ_init = λ .* rand(rng, size(λ))
emissions_init = [
    MultivariatePoissonProcess(λ_init[:, 1]), MultivariatePoissonProcess(λ_init[:, 2])
]

λ_init_log = LogFloat32.(λ_init)
emissions_init_log = [
    MultivariatePoissonProcess(λ_init_log[:, 1]),
    MultivariatePoissonProcess(λ_init_log[:, 2]),
]

hmm_inits = (
    HMM(p0_init, P_init, emissions_init), HMM(p0_init_log, P_init_log, emissions_init_log)
)

for hmm_init in hmm_inits
    hmm_ests = (
        baum_welch_multiple_sequences(obs_sequences, hmm_init;)[1],
        baum_welch_log_multiple_sequences(obs_sequences, hmm_init;)[1],
    )
    for hmm_est in hmm_ests
        p0_est = initial_distribution(hmm_est)
        P_est = transition_matrix(hmm_est)
        λ_est = reduce(hcat, emission_distribution(hmm_est, s).λ for s in 1:2)

        P_error_init = mean(abs, P_init - P)
        P_error = mean(abs, P_est - P)

        λ_error_init = mean(abs, λ_init - λ)
        λ_error = mean(abs, λ_est - λ)

        l_init = sum(logdensityof(hmm_init, obs_sequence) for obs_sequence in obs_sequences)
        l_est = sum(logdensityof(hmm_est, obs_sequence) for obs_sequence in obs_sequences)

        @test typeof(hmm_est) == typeof(hmm_init)
        @test P_error < P_error_init / 5
        @test λ_error < λ_error_init / 5
        @test l_est > l_init
    end
end

end
