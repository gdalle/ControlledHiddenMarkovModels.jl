# # Continuous Markov chain

using HiddenMarkovModels
using HiddenMarkovModels.LogarithmicNumbers
#md using Plots
using Random
using Statistics
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

# ## Construction

# A [`ContinuousMarkovChain`](@ref) object is built by combining a vector of initial probabilities with a matrix of transition rates.

p0 = [0.3, 0.7]
Q = [-2. 2.; 1. -1.]
mc = ContinuousMarkovChain(p0, Q)

# ## Simulation

# To simulate it, we only need to decide the time interval.

h = rand(rng, mc, 4.2, 420.)

# ## Learning

#=
Based on a history with integer marks, we can fit a `ContinuousMarkovChain` with Maximum Likelihood Estimation (MLE).
=#

mc_mle = fit_mle(ContinuousMarkovChain{Float32, Float32}, h)

# Tests (not included in the docs)  #src

error_mle = mean(abs, intensity_matrix(mc_mle) - intensity_matrix(mc))

@test error_mle < 0.1  #src
