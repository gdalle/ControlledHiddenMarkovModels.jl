# # Discrete Markov chain

using HiddenMarkovModels
using Statistics
using Test  #src
using UnicodePlots

# ## Construction

# A [`DiscreteMarkovChain`](@ref) object is built by combining a vector of initial probabilities with a transition matrix.

π0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
dmc = DiscreteMarkovChain(π0, P)

@test nb_states(dmc) == 2  #src
@test stationary_distribution(dmc) ≈ [0.2 / (0.1 + 0.2), 0.1 / (0.1 + 0.2)]  #src

# ## Simulation

# To simulate it, we only need to decide how long the sequence should be.

states = rand(dmc, 100);
scatterplot(states, label=nothing, title="Markov chain evolution", xlabel="Time", ylabel="State")

# ## Learning

# Based on a sequence of states, we can fit a `DiscreteMarkovChain` with Maximum Likelihood.

dmc_est_mle = fit_mle(DiscreteMarkovChain{Float32,Float32}, states)

# As we can see, the error on the transition matrix is quite small.

error_mle = mean(abs, transition_matrix(dmc_est_mle) - transition_matrix(dmc))

@test error_mle < 0.1  #src
