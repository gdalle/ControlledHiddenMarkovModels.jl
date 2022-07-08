# # Hidden Markov Model

using Distributions
using ControlledHiddenMarkovModels
using LogarithmicNumbers
using Random
using Statistics
using Test  #src

rng = Random.default_rng()
Random.seed!(rng, 63)

## Gaussian HMM

struct GaussianHMM{R1,R2,R3} <: AbstractHMM
    p0::Vector{R1}
    P::Matrix{R2}
    emissions::Vector{Normal{R3}}
end

CHMMs.emission_type(::GaussianHMM{R1,R2,R3}) where {R1,R2,R3} = Normal{R3}

function CHMMs.fit_emission_from_multiple_sequences(
    hmm::GaussianHMM{R1,R2,R3}, i::Integer, xs, ws
) where {R1,R2,R3}
    μ = zero(R3)
    w_tot = zero(R3)
    for (x, w) in zip(xs, ws)
        for (xₜ, wₜ) in zip(x, w)
            μ += wₜ * xₜ
            w_tot += wₜ
        end
    end
    μ /= w_tot
    return Normal(μ, one(R3))
end

## Simulation

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
em = [Normal(3.0), Normal(-2.0)]
hmm = GaussianHMM(p0, P, em)

obs_sequences = [rand(rng, hmm, rand(1000:2000))[2] for k in 1:5];

## Learning

p0_init = rand_prob_vec(rng, Float32, 2)
P_init = rand_trans_mat(rng, Float32, 2)
em_init = [Normal(1.0), Normal(-1.0)]
hmm_init = GaussianHMM(p0_init, P_init, em_init)

hmm_est, logL_evolution = baum_welch_multiple_sequences(
    hmm_init, obs_sequences; max_iterations=100, tol=1e-5
);

## Testing

transition_error_init = mean(abs, transition_matrix(hmm_init) - transition_matrix(hmm))
μ_error_init = mean(abs, [emissions(hmm_init)[s].μ - emissions(hmm)[s].μ for s in 1:2])

transition_error = mean(abs, transition_matrix(hmm_est) - transition_matrix(hmm))
μ_error = mean(abs, [emissions(hmm_est)[s].μ - emissions(hmm)[s].μ for s in 1:2])

@test transition_error < transition_error_init / 3  #src
@test μ_error < μ_error_init / 3  #src

## Poisson HMM

struct PoissonHMM{R1,R2,R3} <: AbstractHMM
    p0::Vector{R1}
    P::Matrix{R2}
    emissions::Vector{MultivariatePoissonProcess{R3}}
end

function CHMMs.emission_type(::PoissonHMM{R1,R2,R3}) where {R1,R2,R3}
    return MultivariatePoissonProcess{R3}
end

function CHMMs.fit_emission_from_multiple_sequences(
    hmm::PoissonHMM{R1,R2,R3}, i::Integer, xs, ws
) where {R1,R2,R3}
    event_count = zeros(R3, length(hmm.emissions[i]))
    duration = zero(R3)
    for (x, w) in zip(xs, ws)
        for (xₜ, wₜ) in zip(x, w)
            for m in event_marks(xₜ)
                event_count[m] += wₜ
            end
            duration += wₜ
        end
    end
    λ = event_count ./ duration
    return MultivariatePoissonProcess(λ)
end

## Simulation

em_poisson = [
    MultivariatePoissonProcess([1.0, 2.0, 3.0]), MultivariatePoissonProcess([3.0, 2.0, 1.0])
]
hmm_poisson = PoissonHMM(p0, P, em_poisson)

obs_sequences_poisson = [rand(rng, hmm_poisson, rand(1000:2000))[2] for k in 1:5];

## Learning

em_init_poisson = [
    MultivariatePoissonProcess(rand(rng, 3) .* [1, 2, 3]),
    MultivariatePoissonProcess(rand(rng, 3) .* [3, 2, 1]),
]
hmm_init_poisson = PoissonHMM(p0_init, P_init, em_init_poisson)

hmm_est_poisson, logL_evolution_poisson = baum_welch_multiple_sequences(
    hmm_init_poisson, obs_sequences_poisson; max_iterations=100, tol=1e-5
);

## Testing

transition_error_init_poisson = mean( #src
    abs,  #src
    transition_matrix(hmm_init_poisson) - transition_matrix(hmm_poisson), #src
) #src
λ_error_init_poisson = mean( #src
    abs,  #src
    [ #src
        emissions(hmm_init_poisson)[s].λ[m] - emissions(hmm_poisson)[s].λ[m] #src
        for s in 1:2 for m in 1:3 #src
    ], #src
) #src

transition_error_poisson = mean( #src
    abs,  #src
    transition_matrix(hmm_est_poisson) - transition_matrix(hmm_poisson), #src
) #src
λ_error_poisson = mean( #src
    abs,  #src
    [ #src
        emissions(hmm_est_poisson)[s].λ[m] - emissions(hmm_poisson)[s].λ[m] #src
        for s in 1:2 for m in 1:3 #src
    ], #src
) #src

@test transition_error_poisson < transition_error_init_poisson / 3  #src
@test λ_error_poisson < λ_error_init_poisson / 3  #src
