using HiddenMarkovModels
using Lux
using Lux.NNlib
using Random

mutable struct NeuralDiscreteMarkovChain{A,B,C,D} <: AbstractControlledDiscreteMarkovChain
    p0::A
    P_model::B
    P_params::C
    P_state::D
end

function transition_matrix(ndmc::NeuralDiscreteMarkovChain, u)
    (; P_model, P_params, P_state) = ndmc
    P, new_P_state = Lux.apply(P_model, u, P_params, P_state)
    return P
end

S = 3
U = 10

rng = Random.default_rng()
Random.seed!(rng, 0)

p0 = ones(S) / S

function reshape_square(x::AbstractVector)
    n = isqrt(length(x))
    return reshape(x, n, n)
end

function make_stochastic(x::AbstractMatrix)
    return softmax(x; dims=2)
end

P_model = Chain(Dense(U, S^2), reshape_square, make_stochastic)
P_params, P_state = Lux.setup(rng, P_model)

ndmc = NeuralDiscreteMarkovChain(p0, P_model, P_params, P_state)

TM = transition_matrix(ndmc, rand(U))
