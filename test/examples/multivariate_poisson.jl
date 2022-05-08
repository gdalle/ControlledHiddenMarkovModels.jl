# # Multivariate Poisson process

using HiddenMarkovModels
using LogarithmicNumbers
using Statistics
using Test  #src
using UnicodePlots

# ## Construction

# A [`MultivariatePoissonProcess`](@ref) object is built from a vector of positive event rates.

λ = rand(5)
pp = MultivariatePoissonProcess(λ)

# ## Simulation

# Since it is a temporal point process, we can simulate it on an arbitrary real interval.

history = rand(pp, 3.14, 42.0)

# Each event is defined by a time and an integer mark, which means we can visualize the history in 2 dimensions:

scatterplot(
    event_times(history),
    event_marks(history);
    title="Event history",
    xlabel="Time",
    ylabel="Mark",
)

# ## Learning

# Parameters can learned with Maximum Likelihood Estimation (MLE):

pp_est = fit_mle(MultivariatePoissonProcess{Float64}, history)

# Let's see how well we did

error = mean(abs, pp_est.λ - pp.λ)

# Tests (not included in the docs)  #src

@test error < 0.1  #src
