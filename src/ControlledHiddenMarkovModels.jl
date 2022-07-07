module ControlledHiddenMarkovModels

const CHMMs = ControlledHiddenMarkovModels

using Base.Threads: @threads
using DensityInterface: DensityInterface, densityof, logdensityof
using Distributions: Distributions
using Distributions: Categorical, Exponential, Normal, Poisson, Uniform
using Distributions: suffstats, fit, fit_mle
using LinearAlgebra
using ProgressMeter: Progress, next!
using Random: AbstractRNG, GLOBAL_RNG, rand

## Utilities

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")
include("utils/suffstats.jl")
include("utils/storage.jl")

## Point processes

include("poisson/history.jl")
include("poisson/abstract_poisson_process.jl")
include("poisson/simulation.jl")
include("poisson/density.jl")

include("poisson/multivariate/multivariate_poisson_process.jl")
include("poisson/multivariate/suffstats.jl")
include("poisson/multivariate/prior.jl")
include("poisson/multivariate/fit.jl")

## Markov chains

include("markov/free/markov_chain.jl")
include("markov/free/simulation.jl")
include("markov/free/density.jl")
include("markov/free/suffstats.jl")
include("markov/free/prior.jl")
include("markov/free/fit.jl")

include("markov/controlled/abstract_controlled_mc.jl")
include("markov/controlled/simulation.jl")
include("markov/controlled/density.jl")
include("markov/controlled/predict.jl")

## Hidden Markov Models

include("hmm/free/hidden_markov_model.jl")
include("hmm/free/simulation.jl")
include("hmm/free/obs_density.jl")
include("hmm/free/forward_backward.jl")
include("hmm/free/baum_welch.jl")

include("hmm/controlled/abstract_controlled_hmm.jl")
include("hmm/controlled/simulation.jl")
include("hmm/controlled/density.jl")

export CHMMs

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat

export History
export event_times, event_marks

export AbstractPoissonProcess
export log_intensity, ground_intensity, mark_distribution
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior

export MarkovChain, MarkovChainPrior, stationary_distribution
export nb_states, initial_distribution, transition_matrix
export sample_hitting_times
export AbstractControlledMarkovChain

export HiddenMarkovModel, HMM
export emissions
export baum_welch_multiple_sequences, baum_welch
export AbstractControlledHiddenMarkovModel, AbstractControlledHMM
export emission_parameters, emission_from_parameters
export transition_matrix_and_emission_parameters

export fit, fit_mle, fit_map
export suffstats
export densityof, logdensityof
export flat_prior

end
