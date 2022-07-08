# # Multivariate Poisson process

using ControlledHiddenMarkovModels
using Random
using Statistics
using Test  #src

rng = Random.default_rng()
Random.seed!(rng, 63)

λ = rand(5)
pp = MultivariatePoissonProcess(λ)

history = rand(rng, pp, 3.14, 314.0)

pp_est = fit_mle(typeof(pp), history)

error = mean(abs, pp_est.λ - pp.λ)

@test error < 0.1  #src
