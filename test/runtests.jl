using Aqua
using Documenter
using HiddenMarkovModels
using Random
using Test

Random.seed!(1)

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(HiddenMarkovModels; ambiguities=false)
    end
    @testset verbose = true "Doctests" begin
        doctest(HiddenMarkovModels)
    end
    @testset verbose = true "Examples" begin
        @testset verbose = true "Discrete Markov chain" begin
            include("examples/discrete_markov.jl")
        end
        @testset verbose = true "Continuous Markov chain" begin
            include("examples/continuous_markov.jl")
        end
        @testset verbose = true "Multivariate Poisson process" begin
            include("examples/multivariate_poisson.jl")
        end
        @testset verbose = true "Hidden Markov Model" begin
            include("examples/hmm.jl")
        end
        @testset verbose = true "Controlled HMM" begin
            include("examples/controlled.jl")
        end
    end
end
