# # Hidden Markov Model

using Distributions
using HiddenMarkovModels
using Statistics
using Test  #src

#-

tr = DiscreteMarkovChain(; π0=[0.3, 0.7], P=[0.9 0.1; 0.2 0.8])

#-

em1 = Normal(rand(), 0.5)
em2 = Normal(-rand(), 0.5)
hmm = HiddenMarkovModel(tr, [em1, em2])

#-

observation_sequences = [rand(hmm, rand(500:1000))[2] for k in 1:5];

#-

tr_init = DiscreteMarkovChain(; π0=randprobvec(2), P=randtransmat(2))
em1_init = Normal(1, 1)
em2_init = Normal(-1, 1)
hmm_init = HiddenMarkovModel(tr_init, [em1_init, em2_init])

#-

hmm_est1, logL_evolution1 = baum_welch_multiple_sequences(
    hmm_init, observation_sequences; iterations=100
);
hmm_est2, logL_evolution2 = baum_welch_multiple_sequences_log(
    hmm_init, observation_sequences; iterations=100
);

#-

transition_error1 = mean(abs, transition_matrix(hmm_est1) - transition_matrix(hmm))
transition_error2 = mean(abs, transition_matrix(hmm_est2) - transition_matrix(hmm))

#-

μ_error1 = mean(abs, [emission(hmm_est1, s).μ - emission(hmm, s).μ for s in 1:2])
μ_error2 = mean(abs, [emission(hmm_est2, s).μ - emission(hmm, s).μ for s in 1:2])

#-

σ_error1 = mean(abs, [emission(hmm_est1, s).σ - emission(hmm, s).σ for s in 1:2])
σ_error2 = mean(abs, [emission(hmm_est2, s).σ - emission(hmm, s).σ for s in 1:2])

#-

@test transition_error1 < 0.2  #src
@test transition_error2 < 0.2  #src
@test μ_error1 < 0.1  #src
@test μ_error2 < 0.1  #src
@test σ_error1 < 0.1  #src
@test σ_error2 < 0.1  #src
