```@meta
EditURL = "<unknown>/test/hmm.jl"
```

# Hidden Markov Model

````@example hmm
using Distributions
using HiddenMarkovModels
using LogarithmicNumbers
using Statistics
````

## Construction

A [`HiddenMarkovModel`](@ref) object is build by combining a transition structure (of type [`DiscreteMarkovChain`](@ref)) with a list of emission distributions.

````@example hmm
π0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
transitions = DiscreteMarkovChain(π0, P)
````

````@example hmm
emission1 = Normal(0.4, 0.7)
emission2 = Normal(-0.8, 0.3)
emissions = [emission1, emission2]
````

````@example hmm
hmm = HiddenMarkovModel(transitions, emissions)
````

## Simulation

The simulation utility returns both the sequence of states and the sequence of observations.

````@example hmm
state_sequence, obs_sequence = rand(hmm, 10)
````

With the learning step in mind, we want to generate multiple observations sequences of various lengths.

````@example hmm
obs_sequences = [rand(hmm, rand(200:1000))[2] for k in 1:5];
nothing #hide
````

## Learning

The Baum-Welch algorithm for estimating HMM parameters requires an initial guess, which we choose arbitrarily.
Initial parameters can be created with reduced precision to speed up estimation.

````@example hmm
hmm_init = HiddenMarkovModel(
    DiscreteMarkovChain(rand_prob_vec(Float32, 2), rand_trans_mat(Float32, 2)),
    [Normal(one(Float32)), Normal(-one(Float32))],
)
````

We can now apply the algorithm by setting a tolerance on the loglikelihood increase, as well as a maximum number of iterations.

````@example hmm
hmm_est, logL_evolution = baum_welch_multiple_sequences(
    hmm_init, obs_sequences; max_iterations=1000, tol=1e-5, plot=true
);
nothing #hide
````

As we can see on the plot, each iteration increases the loglikelihood of the estimate: it is a fundamental property of the EM algorithm and its variants.

To improve numerical stability, we can apply the algorithm directly in log scale thanks to [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl).

````@example hmm
hmm_init_log = HiddenMarkovModel(
    DiscreteMarkovChain(rand_prob_vec(LogFloat64, 2), rand_trans_mat(LogFloat64, 2)),
    [Normal(one(LogFloat64)), Normal(-one(LogFloat64))],
)

hmm_est_log, logL_evolution_log = baum_welch_multiple_sequences(
    hmm_init_log, obs_sequences; max_iterations=1000, tol=1e-5, plot=true
);
nothing #hide
````

## Checking results

Let us now compute the estimation error on various parameters.

````@example hmm
transition_error_init = mean(abs, transition_matrix(hmm_init) - transition_matrix(hmm))
μ_error_init = mean(abs, [emission(hmm_init, s).μ - emission(hmm, s).μ for s in 1:2])
σ_error_init = mean(abs, [emission(hmm_init, s).σ - emission(hmm, s).σ for s in 1:2])
(transition_error_init, μ_error_init, σ_error_init)
````

````@example hmm
transition_error = mean(abs, transition_matrix(hmm_est) - transition_matrix(hmm))
μ_error = mean(abs, [emission(hmm_est, s).μ - emission(hmm, s).μ for s in 1:2])
σ_error = mean(abs, [emission(hmm_est, s).σ - emission(hmm, s).σ for s in 1:2])
(transition_error, μ_error, σ_error)
````

As we can see, all of these errors are much smaller than those of `hmm_init`: mission accomplished! The same goes for the logarithmic version.

````@example hmm
transition_error_init_log = mean(
    float ∘ abs, transition_matrix(hmm_init_log) - transition_matrix(hmm)
)
μ_error_init_log = mean(
    float ∘ abs, [emission(hmm_init_log, s).μ - emission(hmm, s).μ for s in 1:2]
)
σ_error_init_log = mean(
    float ∘ abs, [emission(hmm_init_log, s).σ - emission(hmm, s).σ for s in 1:2]
)

(transition_error_init_log, μ_error_init_log, σ_error_init_log)
````

````@example hmm
transition_error_log = mean(
    float ∘ abs, transition_matrix(hmm_est_log) - transition_matrix(hmm)
)
μ_error_log = mean(
    float ∘ abs, [emission(hmm_est_log, s).μ - emission(hmm, s).μ for s in 1:2]
)
σ_error_log = mean(
    float ∘ abs, [emission(hmm_est_log, s).σ - emission(hmm, s).σ for s in 1:2]
)

(transition_error_log, μ_error_log, σ_error_log)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

