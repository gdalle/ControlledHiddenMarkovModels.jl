module HiddenMarkovModels

using DensityInterface
using Distributions
using LinearAlgebra
using LogarithmicNumbers
using LogExpFunctions
using ProgressMeter
using Random: Random, AbstractRNG, GLOBAL_RNG

include("markov/discrete_time/types.jl")
include("markov/discrete_time/simulation.jl")
include("markov/discrete_time/logdensity.jl")
include("markov/discrete_time/inference.jl")

include("hmm/discrete_time/types.jl")
include("hmm/discrete_time/simulation.jl")
include("hmm/discrete_time/forward_backward.jl")
include("hmm/discrete_time/forward_backward_log.jl")
include("hmm/discrete_time/baum_welch.jl")

include("utils/randvals.jl")
include("utils/overflow.jl")

export fit, fit_mle, fit_map

export DiscreteMarkovChain, DiscreteMarkovChainPrior
export nb_states, initial_distribution, transition_matrix, stationary_distribution

export HiddenMarkovModel, HMM
export HiddenMarkovModelPrior, HMMPrior
export transitions
export emission, emissions

export baum_welch, baum_welch_multiple_sequences, baum_welch_multiple_sequences_log

export randprobvec, randtransmat

end
