module ControlledHiddenMarkovModels

const CHMMs = ControlledHiddenMarkovModels

using DensityInterface: DensityInterface, densityof, logdensityof
using Distributions: Distributions
using Distributions: Categorical, Exponential, Normal, Poisson, Product, Uniform
using Distributions: fit, fit_mle, suffstats, product_distribution
using LinearAlgebra
using ProgressLogging: @progress
using Random: AbstractRNG, GLOBAL_RNG, rand

## Utilities

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")
include("utils/storage.jl")
include("utils/logsumexp.jl")

## Point processes

include("poisson/history.jl")
include("poisson/abstract_poisson_process.jl")
include("poisson/simulation.jl")
include("poisson/density.jl")
include("poisson/delimited.jl")

include("poisson/multivariate/multivariate_poisson_process.jl")
include("poisson/multivariate/suffstats.jl")
include("poisson/multivariate/prior.jl")
include("poisson/multivariate/fit.jl")

include("poisson/marked/marked_poisson_process.jl")

## Markov chains

include("markov/free/markov_chain.jl")
include("markov/free/simulation.jl")
include("markov/free/density.jl")
include("markov/free/inference.jl")
include("markov/free/suffstats.jl")
include("markov/free/prior.jl")
include("markov/free/fit.jl")

include("markov/controlled/abstract_controlled_mc.jl")
include("markov/controlled/simulation.jl")
include("markov/controlled/density.jl")
include("markov/controlled/inference.jl")

## Hidden Markov Models

include("hmm/free/abstract_hmm.jl")
include("hmm/free/simulation.jl")
include("hmm/free/obs_density.jl")
include("hmm/free/forward_backward.jl")
include("hmm/free/light_forward.jl")
include("hmm/free/density.jl")
include("hmm/free/inference.jl")
include("hmm/free/baum_welch.jl")

include("hmm/controlled/abstract_controlled_hmm.jl")
include("hmm/controlled/simulation.jl")
include("hmm/controlled/forward_backward.jl")
include("hmm/controlled/light_forward.jl")
include("hmm/controlled/density.jl")
include("hmm/controlled/inference.jl")

export CHMMs

export fit, fit_mle, suffstats
export fit_map
export densityof, logdensityof
export flat_prior

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat
export make_row_stochastic, make_column_stochastic

export History
export event_times, event_marks

export AbstractPoissonProcess
export log_intensity, ground_intensity, mark_distribution
export DelimitedPoissonProcess
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior
export MarkedPoissonProcess

export MarkovChain, MarkovChainPrior, stationary_distribution
export nb_states, initial_distribution, transition_matrix
export sample_hitting_times

export AbstractControlledMarkovChain

export AbstractHiddenMarkovModel, AbstractHMM
export emissions, emission_type, fit_emission_from_multiple_sequences
export baum_welch_multiple_sequences, baum_welch
export infer_current_state

export AbstractControlledHiddenMarkovModel, AbstractControlledHMM
export emission_parameters, emission_from_parameters
export transition_matrix_and_emission_parameters

end
