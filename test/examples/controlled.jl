using HiddenMarkovModels
using HiddenMarkovModels.Distributions
using HiddenMarkovModels.ProgressMeter

using HiddenMarkovModels.Lux
using HiddenMarkovModels.Lux.Optimisers
using HiddenMarkovModels.Lux.NNlib
using HiddenMarkovModels.Lux.Zygote

using Random

#-

rng = Random.default_rng()
Random.seed!(rng)

S = 3
U = 10

#-

struct NeuralGaussianControlledHMM{M} <: AbstractHMM
    p0::Vector{Float64}
    model::M
end

HMMs.nb_states(hmm::NeuralGaussianControlledHMM) = length(hmm.p0)
HMMs.initial_distribution(hmm::NeuralGaussianControlledHMM) = hmm.p0

function HMMs.transition_matrix(hmm::NeuralGaussianControlledHMM, u, ps, st)
    (P, μ), st = Lux.apply(hmm.model, u, ps, st)
    return P
end

function HMMs.emission_distribution(hmm::NeuralGaussianControlledHMM, i::Integer, u, ps, st)
    (P, μ), st = Lux.apply(hmm.model, u, ps, st)
    em_dist = Normal(μ[i])
    return em_dist
end

#-

function make_square(x::AbstractVector)
    S = isqrt(length(x))
    return reshape(x, S, S)
end

make_stochastic(x::AbstractMatrix) = softmax(x; dims=2)

p0 = ones(S) / S

model = Chain(
    Dense(U, 1),
    BranchLayer(Chain(Dense(1, S^2), make_square, make_stochastic), Dense(1, S)),
)

#-

hmm = NeuralGaussianControlledHMM(p0, model)

ps_real, st_real = Lux.setup(rng, hmm.model)
ps, st = Lux.setup(rng, hmm.model)

Lux.apply(hmm.model, rand(U), ps, st)

#-

T = 1000
control_sequence = [randn(U) for t in 1:T];

state_sequence, obs_sequence = rand(rng, hmm, T, control_sequence, ps_real, st_real)

#-

baum_welch(hmm, obs_sequence, control_sequence, ps, st)
