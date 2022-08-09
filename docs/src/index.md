```@meta
CurrentModule = ControlledHiddenMarkovModels
```

# ControlledHiddenMarkovModels.jl

Welcome to the documentation of [ControlledHiddenMarkovModels.jl](https://github.com/gdalle/ControlledHiddenMarkovModels.jl), a package for Hidden Markov Models with exogenous control variables.

## Why would you need it?

This package focuses on discrete-time HMMs with a finite state space, but it's not the only one!
A few alternatives that I am aware of:

- [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl)
- [HMMGradients.jl](https://github.com/idiap/HMMGradients.jl)
- [MarkovModels.jl](https://github.com/FAST-ASR/MarkovModels.jl)
- [Mitosis.jl](https://github.com/mschauer/Mitosis.jl)

I started my own package because I needed specific features that were not simultaneously available elsewhere (to the best of my knowledge):

- Control variables (obviously)
- Compatibility with generic emissions (beyond [Distributions.jl](https://github.com/JuliaStats/Distributions.jl))
- Numerical stability thanks to log-scale computations
- Compatibility with automatic differentiation of parameters
  - in forward mode
  - in reverse mode (WIP)

## Getting started

To install the package, open a Julia [Pkg REPL](https://pkgdocs.julialang.org/v1/getting-started/) and run
```julia
pkg> add https://github.com/gdalle/ControlledHiddenMarkovModels.jl
```

## Mathematical background

To understand the algorithms implemented here, check out the following literature:

> [_A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition_](https://web.mit.edu/6.435/www/Rabiner89.pdf), Lawrence R. Rabiner (1989)

> [_An Input Output HMM Architecture_](https://proceedings.neurips.cc/paper/1994/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf), Yoshua Bengio and Paolo Frasconi (1994)
