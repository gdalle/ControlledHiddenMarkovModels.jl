# # Discrete Markov chain

using LogarithmicNumbers
using ControlledHiddenMarkovModels
using Random
using Statistics
using Test  #src

rng = Random.default_rng()
Random.seed!(rng, 63)

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
mc = MarkovChain(p0, P)

state_sequence = rand(rng, mc, 1000);

mc_mle = fit_mle(MarkovChain{Float32,Float32}, state_sequence)

error_mle = mean(abs, transition_matrix(mc_mle) - transition_matrix(mc))

p0_α = Float32.(1 .+ 4 * [0.5, 0.5])
P_α = Float32.(1 .+ 4 * 10 * [0.5 0.5; 0.5 0.5])
mc_prior = MarkovChainPrior(p0_α, P_α)

mc_map = fit_map(MarkovChain{Float32,Float32}, mc_prior, state_sequence)

transition_matrix(mc_map) - transition_matrix(mc_mle)

mc_mle_log = fit_mle(MarkovChain{LogFloat32,LogFloat32}, state_sequence)

error_mle_log = mean(abs, transition_matrix(mc_mle_log) - transition_matrix(mc))

@test nb_states(mc) == 2  #src
@test stationary_distribution(mc) ≈ [0.2 / (0.1 + 0.2), 0.1 / (0.1 + 0.2)]  #src

@test error_mle < 0.1  #src
@test error_mle_log < 0.1  #src

@test sign.(transition_matrix(mc_map) - transition_matrix(mc_mle)) == [-1 1; 1 -1]  #src
