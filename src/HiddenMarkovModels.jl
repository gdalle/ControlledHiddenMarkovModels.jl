module HiddenMarkovModels

const HMMs = HiddenMarkovModels

using Base.Threads
using DensityInterface
using Distributions
using LinearAlgebra
using LogarithmicNumbers
using ProgressMeter
using Random: Random, AbstractRNG, GLOBAL_RNG

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")

include("point_processes/history.jl")

include("point_processes/multivariate_poisson/types.jl")
include("point_processes/multivariate_poisson/simulation.jl")
include("point_processes/multivariate_poisson/density.jl")
include("point_processes/multivariate_poisson/learning.jl")

include("markov/discrete_time/types.jl")
include("markov/discrete_time/simulation.jl")
include("markov/discrete_time/density.jl")
include("markov/discrete_time/learning.jl")

include("hmm/suffstats.jl")
include("hmm/discrete_time/types.jl")
include("hmm/discrete_time/simulation.jl")
include("hmm/discrete_time/obs_density.jl")
include("hmm/discrete_time/forward_backward.jl")
include("hmm/discrete_time/baum_welch.jl")

export HMMs

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat

export History
export event_times, event_marks

export MultivariatePoissonProcess, MultivariatePoissonProcessPrior

export DiscreteMarkovChain, DiscreteMarkovChainPrior
export nb_states, initial_distribution, transition_matrix, stationary_distribution

export HiddenMarkovModel, HMM
export HiddenMarkovModelPrior, HMMPrior
export get_transitions, get_emission, get_emissions

export baum_welch_multiple_sequences, baum_welch

export fit, fit_mle, fit_map
export suffstats
export densityof, logdensityof
export flat_prior

end
