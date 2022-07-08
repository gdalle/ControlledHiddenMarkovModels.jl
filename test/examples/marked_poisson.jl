# # Multivariate Poisson process

using LogarithmicNumbers
using ControlledHiddenMarkovModels
using Statistics
using Test  #src

rng = Random.default_rng()
Random.seed!(rng, 63)

M = 3
D = 4

λ = LogFloat32(1.)
mark_probs = rand(rng, LogFloat32, D, M)
pp = MarkedPoissonProcess(λ, mark_probs)

history = rand(rng, pp, 0., 1000.)

logdensityof(pp, history)
