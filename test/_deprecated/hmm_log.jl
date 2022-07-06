
# To improve numerical stability, we can apply the algorithm directly in log scale thanks to [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl).

p0_init_log = rand_prob_vec(LogFloat32, 2)
P_init_log = rand_trans_mat(LogFloat32, 2)
transitions_init_log = MarkovChain(p0_init_log, P_init_log)
emissions_init_log = [Normal(one(LogFloat64)), Normal(-one(LogFloat64))]

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

transition_error_init_log = mean( #src
    float ∘ abs,  #src
    transition_matrix(hmm_init_log) - transition_matrix(hmm), #src
) #src
μ_error_init_log = mean( #src
    float ∘ abs,  #src
    [get_emission(hmm_init_log, s).μ - get_emission(hmm, s).μ for s in 1:2], #src
) #src
σ_error_init_log = mean( #src
    float ∘ abs,  #src
    [get_emission(hmm_init_log, s).σ - get_emission(hmm, s).σ for s in 1:2], #src
) #src

transition_error_log = mean( #src
    float ∘ abs,  #src
    transition_matrix(hmm_est_log) - transition_matrix(hmm), #src
) #src
μ_error_log = mean( #src
    float ∘ abs,  #src
    [get_emission(hmm_est_log, s).μ - get_emission(hmm, s).μ for s in 1:2], #src
) #src
σ_error_log = mean( #src
    float ∘ abs,  #src
    [get_emission(hmm_est_log, s).σ - get_emission(hmm, s).σ for s in 1:2], #src
) #src

@test transition_error_log < transition_error_init_log / 3  #src
@test μ_error_log < μ_error_init_log / 3  #src
@test σ_error_log < σ_error_init_log / 3  #src
