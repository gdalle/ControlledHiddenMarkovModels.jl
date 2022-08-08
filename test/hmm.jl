# # Hidden Markov Model

using Distributions
using ControlledHiddenMarkovModels
using LogarithmicNumbers
using PointProcesses
using Random
using Statistics
using Test  #src

rng = Random.default_rng()
Random.seed!(rng, 63)

## Normal HMM

function add_suffstats(stats1::Distributions.NormalStats, stats2::Distributions.NormalStats)
    s = stats1.s + stats2.s
    s2 = stats1.s2 + stats2.s2
    tw = stats1.tw + stats2.tw
    m = s / tw
    stats = Distributions.NormalStats(s, m, s2, tw)
    return stats
end

function CHMMs.fit_from_multiple_sequences(D::Type{<:Normal}, xs, ws)
    stats = reduce(add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws))
    return fit_mle(D, stats)
end

## Simulation

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
μ = [2.0, -3.0]
σ = [0.5, 0.7]
emissions = [Normal(μ[1], σ[1]), Normal(μ[2], σ[2])]
hmm = HMM(p0, P, emissions)

obs_sequences = [rand(rng, hmm, rand(1000:2000))[2] for k in 1:5];

## Learning

p0_init = rand_prob_vec(rng, LogFloat32, 2)
P_init = rand_trans_mat(rng, LogFloat32, 2)
μ_init = [1.0, -1.0]
σ_init = ones(2)
emissions_init = [Normal(μ_init[1], σ_init[1]), Normal(μ_init[2], σ_init[2])]
hmm_init = HMM(p0_init, P_init, emissions_init)

hmm_est, logL_evolution = baum_welch_multiple_sequences(
    obs_sequences, hmm_init; max_iterations=100, tol=1e-5
);

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

l_init = sum(logdensityof(hmm_init, obs_sequence) for obs_sequence in obs_sequences)
l_est = sum(logdensityof(hmm_est, obs_sequence) for obs_sequence in obs_sequences)

@test P_error < P_error_init / 3  #src
@test μ_error < μ_error_init / 3  #src
@test σ_error < σ_error_init / 3  #src
@test l_est > l_init

## Poisson HMM

function CHMMs.fit_from_multiple_sequences(D::Type{<:MultivariatePoissonProcess}, xs, ws)
    stats = reduce(
        PointProcesses.add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws)
    )
    return fit_mle(D, stats)
end

## Simulation

λ = [1.0 3.0; 2.0 2.0; 3.0 1.0]
emissions_poisson = [
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 1]), 0.0, 1.0),
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 2]), 0.0, 1.0),
]
hmm_poisson = HMM(p0, P, emissions_poisson)

obs_sequences_poisson = [rand(rng, hmm_poisson, rand(1000:2000))[2] for k in 1:5];

## Learning

λ_init = λ .* rand(rng, size(λ))
emissions_init_poisson = [
    MultivariatePoissonProcess(λ_init[:, 1]), MultivariatePoissonProcess(λ_init[:, 2])
]
hmm_init_poisson = HMM(p0_init, P_init, emissions_init_poisson)

hmm_est_poisson, logL_evolution_poisson = baum_welch_multiple_sequences(
    obs_sequences_poisson, hmm_init_poisson; max_iterations=100, tol=1e-3
);

p0_est_poisson = initial_distribution(hmm_est_poisson)
P_est_poisson = transition_matrix(hmm_est_poisson)
λ_est = reduce(hcat, emission_distribution(hmm_est_poisson, s).λ for s in 1:2)

## Testing

P_error_init = mean(abs, P_init - P)
P_error_poisson = mean(abs, P_est_poisson - P)

λ_error_init = mean(abs, λ_init - λ)
λ_error = mean(abs, λ_est - λ)

l_init = sum(
    logdensityof(hmm_init_poisson, obs_sequence) for obs_sequence in obs_sequences_poisson
)
l_est = sum(
    logdensityof(hmm_est_poisson, obs_sequence) for obs_sequence in obs_sequences_poisson
)

@test P_error_poisson < P_error_init / 3  #src
@test λ_error < λ_error_init / 3  #src
@test l_est > l_init
