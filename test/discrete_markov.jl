# # Discrete Markov chain

using HiddenMarkovModels
using Statistics
using Test  #src

#-

dmc = DiscreteMarkovChain(; π0=[0.3, 0.7], P=[0.9 0.1; 0.2 0.8])
@test nb_states(dmc) == 2  #src
@test stationary_distribution(dmc) ≈ [0.2 / (0.1 + 0.2), 0.1 / (0.1 + 0.2)]  #src

#-

states = rand(dmc, 1000)

#-

dmc_est_mle = fit_mle(DiscreteMarkovChain, states)

#-

error_mle = mean(abs, transition_matrix(dmc_est_mle) - transition_matrix(dmc))
@test error_mle < 0.1  #src
