```@meta
EditURL = "<unknown>/test/discrete_markov.jl"
```

# Discrete Markov chain

````@example discrete_markov
using HiddenMarkovModels
using LogarithmicNumbers
using Statistics
using UnicodePlots
````

## Construction

A [`DiscreteMarkovChain`](@ref) object is built by combining a vector of initial probabilities with a transition matrix.

````@example discrete_markov
π0 = [0.3, 0.7]
P = [0.9 0.1; 0.2 0.8]
mc = DiscreteMarkovChain(π0, P)
````

## Simulation

To simulate it, we only need to decide how long the sequence should be.

````@example discrete_markov
state_sequence = rand(mc, 100);
scatterplot(
    state_sequence;
    label=nothing,
    title="Markov chain evolution",
    xlabel="Time",
    ylabel="State",
)
````

## Learning

Based on a sequence of states, we can fit a `DiscreteMarkovChain` with Maximum Likelihood Estimation (MLE).
To speed up estimation, we can specify the types of the parameters to estimate, for instance `Float32` instead of `Float64`.

````@example discrete_markov
mc_mle = fit_mle(DiscreteMarkovChain{Float32,Float32}, state_sequence)
````

As we can see, the error on the transition matrix is quite small.

````@example discrete_markov
error_mle = mean(abs, transition_matrix(mc_mle) - transition_matrix(mc))
````

We can also use a Maximum A Posteriori (MAP) approach by specifying a conjugate prior, which contains observed pseudocounts of intializations and transitions.
Let's say we have previously observed 4 trajectories of length 10, with balanced initializations and transitions.

````@example discrete_markov
π0_α = 1 .+ 4 * [0.5, 0.5]
P_α = 1 .+ 4 * 10 * [0.5 0.5; 0.5 0.5]
mc_prior = DiscreteMarkovChainPrior(π0_α, P_α)

mc_map = fit_map(DiscreteMarkovChain{Float32,Float32}, mc_prior, state_sequence)
````

This results in an estimate that puts larger weights on transitions between states $1$ and $2$

````@example discrete_markov
transition_matrix(mc_map) - transition_matrix(mc_mle)
````

Finally, if we fear very small transition probabilities, we can perform the entire estimation in log scale thanks to [LogarithmicNumbers.jl](https://github.com/cjdoris/LogarithmicNumbers.jl).

````@example discrete_markov
mc_mle_log = fit_mle(DiscreteMarkovChain{LogFloat32,LogFloat32}, state_sequence)

error_mle_log = mean(abs, transition_matrix(mc_mle_log) - transition_matrix(mc))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

