```@meta
EditURL = "<unknown>/test/hmm.jl"
```

# Hidden Markov Model

````@example hmm
using Distributions
using HiddenMarkovModels
using Plots
using Statistics
````

## Construction

A [`HiddenMarkovModel`](@ref) object is build by combining a transition structure (typically a [`DiscreteMarkovChain`](@ref)) with a list of emission distributions.

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
states, observations = rand(hmm, 10)
````

With the learning step in mind, we want to generate multiple observations sequences of various lengths.

````@example hmm
observation_sequences = [rand(hmm, rand(300:500))[2] for k in 1:5];
nothing #hide
````

## Learning

The Baum-Welch algorithm for estimating HMM parameters requires an initial guess, which we choose arbitrarily.

````@example hmm
transitions_init = DiscreteMarkovChain(; π0=randprobvec(2), P=randtransmat(2))
emissions_init = [Normal(1, 1), Normal(-1, 1)]
hmm_init = HiddenMarkovModel(transitions_init, emissions_init)
````

We can either use the standard version with a scaling trick...

````@example hmm
hmm_est1, logL_evolution1 = baum_welch_multiple_sequences(
    hmm_init, observation_sequences; iterations=20
);
plot(logL_evolution1, label=nothing, xlabel="Baum-Welch iteration", ylabel="Loglikelihood")
````

````@example hmm
hmm_est1
````

... or the logarithmic version (which is more robust but much slower).

````@example hmm
hmm_est2, logL_evolution2 = baum_welch_multiple_sequences_log(
    hmm_init, observation_sequences; iterations=20
);
plot(logL_evolution2, label=nothing, xlabel="Baum-Welch iteration", ylabel="Loglikelihood")
````

````@example hmm
hmm_est2
````

As we can see on the plots, both procedures increase the loglikelihood of the estimate, as they should.

Let us now compute the estimation error on various parameters.

````@example hmm
transition_error1 = mean(abs, transition_matrix(hmm_est1) - transition_matrix(hmm))
transition_error2 = mean(abs, transition_matrix(hmm_est2) - transition_matrix(hmm))
transition_error1, transition_error2
````

````@example hmm
μ_error1 = mean(abs, [emission(hmm_est1, s).μ - emission(hmm, s).μ for s in 1:2])
μ_error2 = mean(abs, [emission(hmm_est2, s).μ - emission(hmm, s).μ for s in 1:2])
μ_error1, μ_error2
````

````@example hmm
σ_error1 = mean(abs, [emission(hmm_est1, s).σ - emission(hmm, s).σ for s in 1:2])
σ_error2 = mean(abs, [emission(hmm_est2, s).σ - emission(hmm, s).σ for s in 1:2])
σ_error1, σ_error2
````

Since both estimators perform the same operations but on a different scale (standard vs. logarithmic), it is not surprising that their errors coincide up to numerical precision.

More importantly, all of these errors are much smaller than those of `hmm_init`: mission accomplished!

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
