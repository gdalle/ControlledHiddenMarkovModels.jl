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

## Poisson HMM, multiple obs sequences

function CHMMs.fit_mle_from_multiple_sequences(
    D::Type{<:MultivariatePoissonProcess}, xs, ws
)
    stats = reduce(
        PointProcesses.add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws)
    )
    return fit_mle(D, stats)
end

## Simulation

T = 2000
K = 5

λ = [1.0 3.0; 2.0 2.0; 3.0 1.0]
emissions_poisson = [
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 1]), 0.0, 1.0),
    BoundedPointProcess(MultivariatePoissonProcess(λ[:, 2]), 0.0, 1.0),
]
hmm_poisson = HMM(p0, P, emissions_poisson)

obs_sequences_poisson = [rand(rng, hmm_poisson, T)[2] for k in 1:K];

## Learning

λ_init = λ .* rand(rng, size(λ))
emissions_init_poisson = [
    MultivariatePoissonProcess(λ_init[:, 1]), MultivariatePoissonProcess(λ_init[:, 2])
]
hmm_init_poisson = HMM(p0_init, P_init, emissions_init_poisson)

hmm_est_poisson, logL_evolution_poisson = baum_welch_multiple_sequences(
    obs_sequences_poisson, hmm_init_poisson; max_iterations=100, tol=1e-5
);

p0_est_poisson = initial_distribution(hmm_est_poisson)
P_est_poisson = transition_matrix(hmm_est_poisson)
λ_est = reduce(hcat, emission_distribution(hmm_est_poisson, s).λ for s in 1:2)

## Testing

P_error_poisson = mean(abs, P_est_poisson - P)

λ_error_init = mean(abs, λ_init - λ)
λ_error = mean(abs, λ_est - λ)

l_init = sum(
    logdensityof(hmm_init_poisson, obs_sequence) for obs_sequence in obs_sequences_poisson
)
l_est = sum(
    logdensityof(hmm_est_poisson, obs_sequence) for obs_sequence in obs_sequences_poisson
)

@test P_error_poisson < P_error_init / 10
@test λ_error < λ_error_init / 10
@test l_est > l_init
