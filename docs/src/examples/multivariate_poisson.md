```@meta
EditURL = "<unknown>/test/examples/multivariate_poisson.jl"
```

Multivariate Poisson process

````@example multivariate_poisson
using HiddenMarkovModels
using LogarithmicNumbers
using Statistics
using UnicodePlots
````

## Construction

A [`MultivariatePoissonProcess`](@ref) object is built from a vector of positive event rates.

````@example multivariate_poisson
λ = rand(5)
pp = MultivariatePoissonProcess(λ)
````

## Simulation

Since it is a temporal point process, we can simulate it on an arbitrary real interval.

````@example multivariate_poisson
history = rand(pp, 3.14, 42.0)
````

Each event is defined by a time and an integer mark, which means we can visualize the history in 2 dimensions:

````@example multivariate_poisson
scatterplot(
    event_times(history),
    event_marks(history);
    title="Event history",
    xlabel="Time",
    ylabel="Mark",
)
````

## Learning

Parameters can learned with Maximum Likelihood Estimation (MLE):

````@example multivariate_poisson
pp_est = fit_mle(MultivariatePoissonProcess{Float64}, history)
````

Let's see how well we did

````@example multivariate_poisson
error = mean(abs, pp_est.λ - pp.λ)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

