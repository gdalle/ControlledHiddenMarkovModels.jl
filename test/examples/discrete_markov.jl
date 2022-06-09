# # Discrete Markov chain

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

# A [`DiscreteMarkovChain`](@ref) object is built by combining a vector of initial probabilities with a transition matrix.

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
mc = DiscreteMarkovChain(p0, P)

# ## Simulation

# To simulate it, we only need to decide how long the sequence should be.

state_sequence = rand(rng, mc, 1000);

# Let us visualize the sequence of states.

#md scatter(
#md     state_sequence;
#md     title="Markov chain evolution",
#md     xlabel="Time",
#md     ylabel="State",
#md     label=nothing,
#md     margin=5Plots.mm
#md )

# ## Learning

#=
Based on a sequence of states, we can fit a `DiscreteMarkovChain` with Maximum Likelihood Estimation (MLE).
To speed up estimation, we can specify the types of the parameters to estimate, for instance `Float32` instead of `Float64`.
=#

mc_mle = fit_mle(DiscreteMarkovChain{Float32,Float32}, state_sequence)

# As we can see, the error on the transition matrix is quite small.

error_mle = mean(abs, transition_matrix(mc_mle) - transition_matrix(mc))

#=
We can also use a Maximum A Posteriori (MAP) approach by specifying a conjugate prior, which contains observed pseudocounts of intializations and transitions.
Let's say we have previously observed 4 trajectories of length 10, with balanced initializations and transitions.
=#

p0_α = Float32.(1 .+ 4 * [0.5, 0.5])
P_α = Float32.(1 .+ 4 * 10 * [0.5 0.5; 0.5 0.5])
mc_prior = DiscreteMarkovChainPrior(p0_α, P_α)

mc_map = fit_map(DiscreteMarkovChain{Float32,Float32}, mc_prior, state_sequence)

# This results in an estimate that puts larger weights on transitions between states $1$ and $2$

transition_matrix(mc_map) - transition_matrix(mc_mle)

#=
Finally, if we fear very small transition probabilities, we can perform the entire estimation in log scale thanks to [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl).
=#

mc_mle_log = fit_mle(DiscreteMarkovChain{LogFloat32,LogFloat32}, state_sequence)

error_mle_log = mean(abs, transition_matrix(mc_mle_log) - transition_matrix(mc))

# Tests (not included in the docs)  #src

@test nb_states(mc) == 2  #src
@test stationary_distribution(mc) ≈ [0.2 / (0.1 + 0.2), 0.1 / (0.1 + 0.2)]  #src

@test error_mle < 0.1  #src
@test error_mle_log < 0.1  #src

@test sign.(transition_matrix(mc_map) - transition_matrix(mc_mle)) == [-1 1; 1 -1]  #src
