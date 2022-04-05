```@meta
EditURL = "<unknown>/test/discrete_markov.jl"
```

# Discrete Markov chain

````@example discrete_markov
using HiddenMarkovModels
using Plots
using Statistics
````

## Construction

A [`DiscreteMarkovChain`](@ref) object is built by combining a vector of initial probabilities with a transition matrix.

````@example discrete_markov
π0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
dmc = DiscreteMarkovChain(π0, P)
````

## Simulation

To simulate it, we only need to decide how long the sequence should be.

````@example discrete_markov
states = rand(dmc, 100);
scatter(states, label=nothing, xlabel="Time", ylabel="Markov chain state")
````

## Learning

Based on a sequence of states, we can fit a `DiscreteMarkovChain` with Maximum Likelihood.

````@example discrete_markov
dmc_est_mle = fit_mle(DiscreteMarkovChain, states)
````

As we can see, the error on the transition matrix is quite small.

````@example discrete_markov
error_mle = mean(abs, transition_matrix(dmc_est_mle) - transition_matrix(dmc))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

