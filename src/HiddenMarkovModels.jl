module HiddenMarkovModels

using Base.Threads
using DensityInterface
using Distributions
using LinearAlgebra
using LogExpFunctions
using ProgressMeter
using Random: Random, AbstractRNG, GLOBAL_RNG
using UnicodePlots

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")
include("utils/plot.jl")

include("markov/discrete_time/types.jl")
include("markov/discrete_time/simulation.jl")
include("markov/discrete_time/logdensity.jl")
include("markov/discrete_time/learning.jl")

include("hmm/discrete_time/types.jl")
include("hmm/discrete_time/simulation.jl")
include("hmm/discrete_time/forward_backward.jl")
include("hmm/discrete_time/forward_backward_log.jl")
include("hmm/discrete_time/baum_welch.jl")

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat

export DiscreteMarkovChain, DiscreteMarkovChainPrior
export nb_states, initial_distribution, transition_matrix, stationary_distribution

export HiddenMarkovModel, HMM
export HiddenMarkovModelPrior, HMMPrior
export transitions
export emission, emissions

export baum_welch_multiple_sequences, baum_welch

export fit, fit_mle, fit_map

end
