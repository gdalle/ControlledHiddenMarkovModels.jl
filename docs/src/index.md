```@meta
CurrentModule = HiddenMarkovModels
```

# HiddenMarkovModels.jl

Welcome to the documentation of [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl), a lightweight package for working with Markov chains and Hidden Markov Models.

## Getting started

To install the package, open a Julia [Pkg REPL](https://pkgdocs.julialang.org/v1/getting-started/) and run
```julia
pkg> add https://github.com/gdalle/HiddenMarkovModels.jl
```

## Mathematical background

To understand the algorithms implemented here, check out the following literature:

> [_A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition_](https://web.mit.edu/6.435/www/Rabiner89.pdf), Lawrence R. Rabiner (1989)

## Alternatives

If you don't find what you are looking for around here, there are several other Julia packages with a focus on Markovian modeling.
Here are the ones that I am aware of:

- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl)
- [Mitosis.jl](https://github.com/mschauer/Mitosis.jl)
- [ContinuousTimeMarkov.jl](https://github.com/tpapp/ContinuousTimeMarkov.jl)
- [PiecewiseDeterministicMarkovProcesses.jl](https://github.com/rveltz/PiecewiseDeterministicMarkovProcesses.jl)

The reason I implemented my own was because I needed specific features that were not simultaneously available elsewhere (to the best of my knowledge):

- Compatibility with generic emissions that go beyond [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) objects
- Discrete **and** continuous time versions
- MAP estimation with priors
