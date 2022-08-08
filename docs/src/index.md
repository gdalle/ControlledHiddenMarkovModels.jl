```@meta
CurrentModule = ControlledHiddenMarkovModels
```

# ControlledHiddenMarkovModels.jl

Welcome to the documentation of [ControlledHiddenMarkovModels.jl](https://github.com/gdalle/ControlledHiddenMarkovModels.jl), a package for working with Markov chains and Hidden Markov Models that may be influenced by exogenous control variables.

## Getting started

To install the package, open a Julia [Pkg REPL](https://pkgdocs.julialang.org/v1/getting-started/) and run
```julia
pkg> add https://github.com/gdalle/ControlledHiddenMarkovModels.jl
```

## Mathematical background

To understand the algorithms implemented here, check out the following literature:

> [_A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition_](https://web.mit.edu/6.435/www/Rabiner89.pdf), Lawrence R. Rabiner (1989)

> [_An Input Output HMM Architecture_](https://proceedings.neurips.cc/paper/1994/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf), Yoshua Bengio and Paolo Frasconi (1994)

## Alternatives

If you don't find what you are looking for around here, there are several other Julia packages with a focus on Markovian modeling.
Here are the ones that I am aware of:

- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl)
- [ControlledHiddenMarkovModels.jl](https://github.com/BenConnault/ControlledHiddenMarkovModels.jl)
- [Mitosis.jl](https://github.com/mschauer/Mitosis.jl)
- [ContinuousTimeMarkov.jl](https://github.com/tpapp/ContinuousTimeMarkov.jl)
- [PiecewiseDeterministicMarkovProcesses.jl](https://github.com/rveltz/PiecewiseDeterministicMarkovProcesses.jl)

The reason I implemented my own was because I needed specific features that were not simultaneously available elsewhere (to the best of my knowledge):

- Control variables
- Compatibility with generic emissions (beyond [Distributions.jl](https://github.com/JuliaStats/Distributions.jl))
- Numerical stability thanks to log-scale computations
- MAP estimation with priors (WIP)
