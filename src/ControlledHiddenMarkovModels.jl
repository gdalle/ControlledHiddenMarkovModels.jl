"""
    ControlledHiddenMarkovModels

A package for Hidden Markov Models with exogenous control variables.
"""
module ControlledHiddenMarkovModels

"""
    CHMMs

Alias for [`ControlledHiddenMarkovModels`](@ref).
"""
const CHMMs = ControlledHiddenMarkovModels

using DensityInterface: DensityInterface, densityof, logdensityof
using Distributions: Categorical, fit_mle
using LinearAlgebra
using Random: AbstractRNG, GLOBAL_RNG, rand

include("utils/prob_vec.jl")
include("utils/trans_mat.jl")
include("utils/overflow.jl")
include("utils/logsumexp.jl")

include("free/abstract_hmm.jl")
include("free/simulation.jl")
include("free/obs_density.jl")
include("free/storage.jl")
include("free/forward_backward.jl")
include("free/forward_backward_log.jl")
include("free/inference.jl")

include("free/concrete/hmm.jl")
include("free/concrete/re_estimation.jl")
include("free/concrete/baum_welch.jl")

include("controlled/abstract_controlled_hmm.jl")
include("controlled/simulation.jl")
include("controlled/forward_backward.jl")
include("controlled/forward_backward_log.jl")
include("controlled/inference.jl")

export CHMMs

export logdensityof

export is_prob_vec, rand_prob_vec, make_prob_vec!, make_log_prob_vec!
export is_trans_mat, rand_trans_mat, make_trans_mat!, make_log_trans_mat!
export logsumexp

export AbstractHiddenMarkovModel, AbstractHMM
export nb_states
export initial_distribution, log_initial_distribution
export transition_matrix, log_transition_matrix
export emission_distribution
export infer_current_state

export HiddenMarkovModel, HMM
export baum_welch, baum_welch_log
export baum_welch_multiple_sequences, baum_welch_log_multiple_sequences
export emission_type

export AbstractControlledHiddenMarkovModel, AbstractControlledHMM
export emission_parameters

end
