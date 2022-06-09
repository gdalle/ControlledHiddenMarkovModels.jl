using HiddenMarkovModels
using Lux
using Lux.Optimisers
using Lux.NNlib
using Lux.Zygote
using ProgressMeter
using Random

#-

rng = Random.default_rng()
Random.seed!(rng)

#-

mutable struct NeuralDiscreteMarkovChain{A,B} <: AbstractControlledDiscreteMarkovChain
    p0::A
    P_model::B
end

#-

HMMs.initial_distribution(ndmc::NeuralDiscreteMarkovChain) = ndmc.p0

function HMMs.transition_matrix(ndmc::NeuralDiscreteMarkovChain, u, ps, st)
    (; P_model) = ndmc
    P, _ = Lux.apply(P_model, u, ps, st)
    return P
end

#-

function reshape_square(x::AbstractVector)
    n = isqrt(length(x))
    return reshape(x, n, n)
end

function make_stochastic(x::AbstractMatrix)
    # return softmax(x; dims=2)
    p = softplus.(x)
    return p ./ sum(p; dims=2)
end

#-

S = 5
U = 10

p0 = ones(S) / S
P_model = Chain(Dense(U, 1), Dense(1, S^2), reshape_square, make_stochastic)
ndmc = NeuralDiscreteMarkovChain(p0, P_model)

ps_real, st_real = Lux.setup(rng, P_model)
ps, st = Lux.setup(rng, P_model)

#-

T = 1000
control_sequence = [rand(U) for t in 1:T];
state_sequence = rand(rng, ndmc, control_sequence, ps_real, st_real)

logdensityof(ndmc, state_sequence, control_sequence, ps_real, st_real)
logdensityof(ndmc, state_sequence, control_sequence, ps, st)

#-

st_opt = Optimisers.setup(Optimisers.ADAM(), ps)

@showprogress for iteration in 1:100
    gs = first(gradient(
        p -> -logdensityof(ndmc, state_sequence, control_sequence, p, st), ps
    ))
    st_opt, ps = Optimisers.update(st_opt, ps, gs)
end

logdensityof(ndmc, state_sequence, control_sequence, ps_real, st_real)
logdensityof(ndmc, state_sequence, control_sequence, ps, st)
