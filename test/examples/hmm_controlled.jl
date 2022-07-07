using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using Lux
using Optimization
using NNlib
using ProgressMeter
using Random
using Test

using ForwardDiff: ForwardDiff
using OptimizationFlux: OptimizationFlux
using OptimizationOptimJL: OptimizationOptimJL
using Zygote: Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 1
S = 3
E = 2
T = 100

make_stochastic(x) = x ./ sum(x; dims=2)

p0 = rand_prob_vec(S)

## Struct

struct NeuralHMM{R,M} <: AbstractControlledHMM
    p0::Vector{R}
    P_μ_model::M
end

CHMMs.nb_states(nhmm::NeuralHMM) = length(nhmm.p0)
CHMMs.initial_distribution(nhmm::NeuralHMM) = nhmm.p0
CHMMs.transition_matrix(nhmm::NeuralHMM, u, ps, st) = nhmm.P_μ_model(u, ps, st)[1][1]
CHMMs.emission_parameters(nhmm::NeuralHMM, u, ps, st) = nhmm.P_μ_model(u, ps, st)[1][2]

function CHMMs.emission_from_parameters(nhmm::NeuralHMM, μ::AbstractVector)
    return MvNormal(μ, 1)
end

function CHMMs.transition_matrix_and_emission_parameters(nhmm::NeuralHMM, u, ps, st)
    return nhmm.P_μ_model(u, ps, st)[1]
end

P_μ_model = Chain(
    Dense(U, 1),
    BranchLayer(
        Chain(Dense(1, S^2, softplus), ReshapeLayer((S, S)), make_stochastic),
        Chain(Dense(1, S * E), ReshapeLayer((E, S))),
    ),
)
ps, st = Lux.setup(rng, P_μ_model);

nhmm = NeuralHMM(p0, P_μ_model);

control_sequence = ones(U, T);
state_sequence, obs_sequence = rand(nhmm, control_sequence, ps, st);

logdensityof(nhmm, obs_sequence, control_sequence, ps, st)
