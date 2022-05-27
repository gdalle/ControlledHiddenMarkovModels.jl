using Aqua
using Documenter
using HiddenMarkovModels
using Random
using Test

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
        @testset verbose = true "Multivariate Poisson process" begin
            include("examples/multivariate_poisson.jl")
        end
        @testset verbose = true "Hidden Markov Model" begin
            include("examples/hmm.jl")
        end
    end
end
