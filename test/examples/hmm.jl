# # Hidden Markov Model

using Distributions
using HiddenMarkovModels
using LogarithmicNumbers
#md using Plots
using Statistics
using Test  #src

# ## Construction

# A [`HiddenMarkovModel`](@ref) object is build by combining a transition structure (of type [`DiscreteMarkovChain`](@ref)) with a list of emission distributions.

p0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
transitions = DiscreteMarkovChain(p0, P)

#-

emission1 = Normal(0.4, 0.7)
emission2 = Normal(-0.8, 0.3)
emissions = [emission1, emission2]

#-

hmm = HiddenMarkovModel(transitions, emissions)

# ## Simulation

# The simulation utility returns both the sequence of states and the sequence of observations.

state_sequence, obs_sequence = rand(hmm, 10)

# With the learning step in mind, we want to generate multiple observations sequences of various lengths.

obs_sequences = [rand(hmm, rand(200:1000))[2] for k in 1:5];

# ## Learning

#=
The Baum-Welch algorithm for estimating HMM parameters requires an initial guess, which we choose arbitrarily.
Initial parameters can be created with reduced precision to speed up estimation.
=#

p0_init = rand_prob_vec(Float32, 2)
P_init = rand_trans_mat(Float32, 2)
transitions_init = DiscreteMarkovChain(p0_init, P_init)
emissions_init = [Normal(one(Float32)), Normal(-one(Float32))]

hmm_init = HiddenMarkovModel(transitions_init, emissions_init)

# We can now apply the algorithm by setting a tolerance on the loglikelihood increase, as well as a maximum number of iterations.

hmm_est, logL_evolution = baum_welch_multiple_sequences(
    hmm_init, obs_sequences; max_iterations=100, tol=1e-5
);

# As we can see on the plot, each iteration increases the loglikelihood of the estimate: it is a fundamental property of the EM algorithm and its variants.

#md plot(
#md     logL_evolution;
#md     title="Baum-Welch convergence (Normal emissions)",
#md     xlabel="Iteration",
#md     ylabel="Log-likelihood",
#md     label=nothing,
#md     margin=5Plots.mm
#md )

# To improve numerical stability, we can apply the algorithm directly in log scale thanks to [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl).

p0_init_log = rand_prob_vec(LogFloat32, 2)
P_init_log = rand_trans_mat(LogFloat32, 2)
transitions_init_log = DiscreteMarkovChain(p0_init_log, P_init_log)
emissions_init_log = [Normal(one(LogFloat32)), Normal(-one(LogFloat32))]

hmm_init_log = HiddenMarkovModel(transitions_init_log, emissions_init_log)

hmm_est_log, logL_evolution_log = baum_welch_multiple_sequences(
    hmm_init_log, obs_sequences; max_iterations=100, tol=1e-5
);

#md plot(
#md     logL_evolution_log;
#md     title="Log Baum-Welch convergence (Normal emissions)",
#md     xlabel="Iteration",
#md     ylabel="Log-likelihood",
#md     label=nothing,
#md     margin=5Plots.mm
#md )

# Let us now compute the estimation error on various parameters.

transition_error_init = mean(abs, transition_matrix(hmm_init) - transition_matrix(hmm))
μ_error_init = mean(abs, [emission(hmm_init, s).μ - emission(hmm, s).μ for s in 1:2])
σ_error_init = mean(abs, [emission(hmm_init, s).σ - emission(hmm, s).σ for s in 1:2])
(transition_error_init, μ_error_init, σ_error_init)

#-

transition_error = mean(abs, transition_matrix(hmm_est) - transition_matrix(hmm))
μ_error = mean(abs, [emission(hmm_est, s).μ - emission(hmm, s).μ for s in 1:2])
σ_error = mean(abs, [emission(hmm_est, s).σ - emission(hmm, s).σ for s in 1:2])
(transition_error, μ_error, σ_error)

# As we can see, all of these errors are much smaller than those of `hmm_init`: mission accomplished! The same goes for the logarithmic version.

# ## Custom emission distributions

#=
One of the major selling points for HiddenMarkovModels.jl is that the user can define their own emission distributions.
Here we give an example where emissions are of type [`MultivariatePoissonProcess`](@ref) with state-dependent rates.
=#

emissions_poisson = [
    MultivariatePoissonProcess([1.0, 2.0, 3.0]), MultivariatePoissonProcess([3.0, 2.0, 1.0])
]

hmm_poisson = HMM(transitions, emissions_poisson)

# We can simulate and learn it using the exact same procedure.

state_sequence_poisson, obs_sequence_poisson = rand(hmm_poisson, 1000);

emissions_init_poisson = [
    MultivariatePoissonProcess([rand(), 2rand(), 3rand()]),
    MultivariatePoissonProcess([3rand(), 2rand(), rand()]),
]

hmm_init_poisson = HMM(transitions_init, emissions_init_poisson)

hmm_est_poisson, logL_evolution_poisson = baum_welch(
    hmm_init_poisson, obs_sequence_poisson; max_iterations=100, tol=1e-5
);

#md plot(
#md     logL_evolution_poisson;
#md     title="Baum-Welch convergence (Poisson emissions)",
#md     xlabel="Iteration",
#md     ylabel="Log-likelihood",
#md     label=nothing,
#md     margin=5Plots.mm
#md )

# Tests (not included in docs) #src

# Log errors  #src

transition_error_init_log = mean( #src
    float ∘ abs,  #src
    transition_matrix(hmm_init_log) - transition_matrix(hmm), #src
) #src
μ_error_init_log = mean( #src
    float ∘ abs,  #src
    [emission(hmm_init_log, s).μ - emission(hmm, s).μ for s in 1:2], #src
) #src
σ_error_init_log = mean( #src
    float ∘ abs,  #src
    [emission(hmm_init_log, s).σ - emission(hmm, s).σ for s in 1:2], #src
) #src

transition_error_log = mean( #src
    float ∘ abs,  #src
    transition_matrix(hmm_est_log) - transition_matrix(hmm), #src
) #src
μ_error_log = mean( #src
    float ∘ abs,  #src
    [emission(hmm_est_log, s).μ - emission(hmm, s).μ for s in 1:2], #src
) #src
σ_error_log = mean( #src
    float ∘ abs,  #src
    [emission(hmm_est_log, s).σ - emission(hmm, s).σ for s in 1:2], #src
) #src

# Poisson errors  #src

transition_error_init_poisson = mean( #src
    abs,  #src
    transition_matrix(hmm_init_poisson) - transition_matrix(hmm_poisson), #src
) #src
λ_error_init_poisson = mean( #src
    abs,  #src
    [ #src
        emission(hmm_init_poisson, s).λ[m] - emission(hmm_poisson, s).λ[m] #src
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
        emission(hmm_est_poisson, s).λ[m] - emission(hmm_poisson, s).λ[m] #src
        for s in 1:2 for m in 1:3 #src
    ], #src
) #src

# Check that errors went down  #src

@test transition_error < transition_error_init / 3  #src
@test μ_error < μ_error_init / 3  #src
@test σ_error < σ_error_init / 3  #src

@test transition_error_log < transition_error_init_log / 3  #src
@test μ_error_log < μ_error_init_log / 3  #src
@test σ_error_log < σ_error_init_log / 3  #src

@test transition_error_poisson < transition_error_init_poisson / 3  #src
@test λ_error_poisson < λ_error_init_poisson / 3  #src
