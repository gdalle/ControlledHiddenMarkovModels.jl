using Distributions
using ControlledHiddenMarkovModels
using Lux
using NNlib
using Optimisers
using ProgressMeter
using Random
using Zygote

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

S = 3
U = 10

#-

struct NeuralGaussianControlledHMM{M} <: AbstractHMM
    p0::Vector{Float64}
    model::M
end

HMMs.nb_states(hmm::NeuralGaussianControlledHMM) = length(hmm.p0)
HMMs.initial_distribution(hmm::NeuralGaussianControlledHMM) = hmm.p0

function HMMs.transition_matrix_and_emission_distributions(
    hmm::NeuralGaussianControlledHMM, u, ps, st
)
    (P, μ), st = Lux.apply(hmm.model, u, ps, st)
    return P, [Normal(μ[i]) for i in eachindex(μ)]
end

#-

make_square(x::AbstractVector) = reshape(x, isqrt(length(x)), isqrt(length(x)))
make_stochastic(x::AbstractMatrix) = softmax(x; dims=2)

p0 = ones(S) / S

transition_model = Chain(Dense(1, S^2), make_square, make_stochastic)
emission_model = Dense(1, S)
model = Chain(Dense(U, 1), BranchLayer(transition_model, emission_model))

#-

hmm = NeuralGaussianControlledHMM(p0, model)

ps_real, st_real = Lux.setup(rng, hmm.model)
ps, st = Lux.setup(rng, hmm.model)

Lux.apply(hmm.model, rand(U), ps, st)

#-

T = 1000
control_matrix = [randn(U) for t in 1:T];

state_sequence, obs_sequence = rand(rng, hmm, T, control_matrix, ps_real, st_real)

#-

ps_est, logL_evolution = baum_welch(hmm, obs_sequence, control_matrix, ps, st)

@time baum_welch(hmm, obs_sequence, control_matrix, ps, st)
@profview baum_welch(hmm, obs_sequence, control_matrix, ps, st)
