# # Hidden Markov Model

using Distributions
using HiddenMarkovModels
using Statistics
using Test  #src

# ## Construction

# A [`HiddenMarkovModel`](@ref) object is build by combining a transition structure (typically a [`DiscreteMarkovChain`](@ref)) with a list of emission distributions.

π0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
transitions = DiscreteMarkovChain(π0, P)

#-

emission1 = Normal(0.4, 0.7)
emission2 = Normal(-0.8, 0.3)
emissions = [emission1, emission2]

#-

hmm = HiddenMarkovModel(transitions, emissions)

# ## Simulation

# The simulation utility returns both the sequence of states and the sequence of observations.

states, observations = rand(hmm, 10)

# With the learning step in mind, we want to generate multiple observations sequences of various lengths.

observation_sequences = [rand(hmm, rand(300:500))[2] for k in 1:5];

# ## Learning

# The Baum-Welch algorithm for estimating HMM parameters requires an initial guess, which we choose arbitrarily.

transitions_init = DiscreteMarkovChain(
    rand_prob_vec(Float32, 2), rand_trans_mat(Float32, 2)
)
emissions_init = [Normal(one(Float32)), Normal(-one(Float32))]
hmm_init = HiddenMarkovModel(transitions_init, emissions_init)

# We can now apply the algorithm by setting a tolerance on the loglikelihood increase, as well as a maximum number of iterations.

hmm_est, logL_evolution = baum_welch_multiple_sequences(
    hmm_init, observation_sequences; max_iterations=1000, tol=1e-5, plot=true
);

# As we can see on the plots, this variant of the EM algorithm increases the loglikelihood of the estimate, as it should.

# ## Checking results

# Let us now compute the estimation error on various parameters.

transition_error = mean(abs, transition_matrix(hmm_est) - transition_matrix(hmm))

#-

μ_error = mean(abs, [emission(hmm_est, s).μ - emission(hmm, s).μ for s in 1:2])

#-

σ_error = mean(abs, [emission(hmm_est, s).σ - emission(hmm, s).σ for s in 1:2])

# As we can see, all of these errors are much smaller than those of `hmm_init`: mission accomplished!

@test transition_error < 0.1  #src
@test μ_error < 0.1  #src
@test σ_error < 0.1  #src
