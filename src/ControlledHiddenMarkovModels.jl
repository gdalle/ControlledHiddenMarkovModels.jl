module ControlledHiddenMarkovModels

const CHMMs = ControlledHiddenMarkovModels

using DensityInterface: DensityInterface, densityof, logdensityof
using Distributions: Categorical
using LinearAlgebra
using ProgressLogging: @progress
using Random: AbstractRNG, GLOBAL_RNG, rand

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")
include("utils/logsumexp.jl")

include("free/abstract_hmm.jl")
include("free/simulation.jl")
include("free/forward_backward.jl")
include("free/inference.jl")

include("free/concrete/hmm.jl")
include("free/concrete/baum_welch.jl")

# include("controlled/abstract_controlled_hmm.jl")
# include("controlled/simulation.jl")
# include("controlled/light_forward.jl")

export CHMMs

export logdensityof

export is_prob_vec, rand_prob_vec
export is_trans_mat, rand_trans_mat
export logsumexp

export AbstractHiddenMarkovModel, AbstractHMM
export nb_states
export initial_distribution, transition_matrix
export emission_distribution
export infer_current_state

export HiddenMarkovModel, HMM
export baum_welch_multiple_sequences, baum_welch
export emission_type, fit_from_multiple_sequences

# export AbstractControlledHiddenMarkovModel, AbstractControlledHMM
# export emission_parameters, emission_distribution

end
