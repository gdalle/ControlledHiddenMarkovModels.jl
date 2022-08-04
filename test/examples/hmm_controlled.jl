## Imports

using ComponentArrays
using ControlledHiddenMarkovModels
using Distributions
using LinearAlgebra
using Optimization
using ProgressMeter
using Statistics
using Random
using Test
using ThreadsX

using ForwardDiff: ForwardDiff
using OptimizationOptimisers: OptimizationOptimisers

## Setup

rng = Random.default_rng()
Random.seed!(rng, 0)

U = 50
S = 5
T = 100
C = 40
V = 3
K = 10

function callback(x, l)
    @info "Iteration" l
    return false
end

## Controlled Poisson HMM

struct ControlledPoissonHMM <: AbstractControlledHMM
    S::Int
end

CHMMs.nb_states(hmm::ControlledPoissonHMM) = hmm.S

function CHMMs.log_initial_distribution(hmm::ControlledPoissonHMM, parameters)
    return log.(ones(hmm.S) / hmm.S)
end

function CHMMs.log_transition_matrix!(
    logP::Matrix, hmm::ControlledPoissonHMM, control, parameters
)
    shift = mean(control)
    logP .= parameters.logP .+ shift
    @views for s in axes(logP, 1)
        logP[s, :] .-= CHMMs.logsumexp_stream(logP[s, :])
    end
    return logP
end

function CHMMs.log_transition_matrix(
    hmm::ControlledPoissonHMM, control::AbstractVector, parameters
)
    logP = similar(parameters.logP)
    CHMMs.log_transition_matrix!(logP, hmm, control, parameters)
    return logP
end

function CHMMs.emission_parameters!(
    θ::AbstractVector, hmm::ControlledPoissonHMM, control, parameters
)
    (; logλ, logp) = parameters.θ
    shift = mean(control)
    for i in eachindex(θ.logλ)
        θ.logλ[i] = logλ[i] + shift
    end
    for i in eachindex(θ.logp)
        θ.logp[i] = logp[i] + shift
    end
    @views for s in axes(θ.logp, 3), c in axes(θ.logp, 2)
        θ.logp[:, c, s] .-= CHMMs.logsumexp_stream(θ.logp[:, c, s])
    end
    return θ
end

function CHMMs.emission_parameters(hmm::ControlledPoissonHMM, control, parameters)
    θ = similar(parameters.θ)
    CHMMs.emission_parameters!(θ, hmm, control, parameters)
    return θ
end

function CHMMs.emission_distribution(
    hmm::ControlledPoissonHMM, θ::AbstractVector, s::Integer
)
    logλ = θ.logλ[s]
    logp = @view θ.logp[:, :, s]
    return DelimitedPoissonProcess(LogMarkedPoissonProcess(logλ, logp), 0.0, 1.0)
end

## Simulation

hmm = ControlledPoissonHMM(S);

parameters_true = ComponentVector(;
    logP=randn(S, S), θ=ComponentVector(; logλ=randn(S) .+ 2, logp=randn(V, C, S))
);
parameters_init = ComponentVector(;
    logP=randn(S, S), θ=ComponentVector(; logλ=randn(S) .+ 2, logp=randn(V, C, S))
);

control_sequences = [[rand(U) for t in 1:T] for k in 1:K];

obs_sequences = [
    last(rand(hmm, control_sequence, parameters_true)) for
    control_sequence in control_sequences
];

## Learning

data = (hmm, obs_sequences, control_sequences);

function loss(parameters, data)
    (hmm, obs_sequences, control_sequences) = data
    return -ThreadsX.sum(
        logdensityof(hmm, os, cs, parameters) for
        (os, cs) in zip(obs_sequences, control_sequences)
    )
end

f = OptimizationFunction(loss, Optimization.AutoForwardDiff());
prob = OptimizationProblem(f, parameters_init, data);
res = solve(prob, OptimizationOptimisers.Adam(); maxiters=10);

parameters_est = res.u;

## Testing

logL_true = logdensityof(hmm, obs_sequences[1], control_sequences[1], parameters_true)
logL_init = logdensityof(hmm, obs_sequences[1], control_sequences[1], parameters_init)
logL_est = logdensityof(hmm, obs_sequences[1], control_sequences[1], parameters_est)

@test logL_est > logL_init
