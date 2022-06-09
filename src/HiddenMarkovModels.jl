module HiddenMarkovModels

const HMMs = HiddenMarkovModels

using Base.Threads
using DensityInterface
using Distributions
using FillArrays
using LinearAlgebra
using LogarithmicNumbers
using ProgressMeter
using Random

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/rates_mat.jl")
include("utils/overflow.jl")
include("utils/suffstats.jl")

include("point_processes/history.jl")

include("point_processes/multivariate_poisson/concrete.jl")
include("point_processes/multivariate_poisson/simulation.jl")
include("point_processes/multivariate_poisson/density.jl")
include("point_processes/multivariate_poisson/suffstats.jl")
include("point_processes/multivariate_poisson/prior.jl")
include("point_processes/multivariate_poisson/fit.jl")

include("markov/abstract.jl")

include("markov/discrete_time/abstract.jl")
include("markov/discrete_time/density.jl")
include("markov/discrete_time/simulation.jl")

include("markov/discrete_time/free/concrete.jl")
include("markov/discrete_time/free/suffstats.jl")
include("markov/discrete_time/free/prior.jl")
include("markov/discrete_time/free/fit.jl")

include("markov/discrete_time/controlled/abstract.jl")

include("markov/continuous_time/free/abstract.jl")
include("markov/continuous_time/free/concrete.jl")
include("markov/continuous_time/free/simulation.jl")
include("markov/continuous_time/free/density.jl")
include("markov/continuous_time/free/suffstats.jl")
include("markov/continuous_time/free/prior.jl")
include("markov/continuous_time/free/fit.jl")

include("hmm/discrete_time/abstract.jl")
include("hmm/discrete_time/concrete.jl")
include("hmm/discrete_time/prior.jl")
include("hmm/discrete_time/simulation.jl")
include("hmm/discrete_time/obs_density.jl")
include("hmm/discrete_time/forward_backward.jl")
include("hmm/discrete_time/baum_welch.jl")

export HMMs

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat
export is_rates_mat, rand_rates_mat

export History
export event_times, event_marks

export MultivariatePoissonProcess, MultivariatePoissonProcessPrior

export AbstractMarkovChain, AbstractDiscreteMarkovChain, AbstractContinuousMarkovChain
export AbstractControlledDiscreteMarkovChain
export DiscreteMarkovChain, DiscreteMarkovChainPrior
export ContinuousMarkovChain, ContinuousMarkovChainPrior
export nb_states, initial_distribution, stationary_distribution
export transition_matrix, transition_probability
export intensity_matrix, intensity_value

export HiddenMarkovModel, HMM
export HiddenMarkovModelPrior, HMMPrior
export get_transitions, get_emission, get_emissions

export baum_welch_multiple_sequences, baum_welch

export fit, fit_mle, fit_map
export suffstats
export densityof, logdensityof
export flat_prior

end
