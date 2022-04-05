```@meta
EditURL = "<unknown>/test/discrete_markov.jl"
```

# Discrete Markov chain

````@example discrete_markov
using HiddenMarkovModels
using Statistics
````

````@example discrete_markov
dmc = DiscreteMarkovChain(; π0=[0.3, 0.7], P=[0.9 0.1; 0.2 0.8])
````

````@example discrete_markov
states = rand(dmc, 1000)
````

````@example discrete_markov
dmc_est_mle = fit_mle(DiscreteMarkovChain, states)
````

````@example discrete_markov
error_mle = mean(abs, transition_matrix(dmc_est_mle) - transition_matrix(dmc))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

