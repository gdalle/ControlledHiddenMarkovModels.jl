using Aqua
using Documenter
using HiddenMarkovModels
using Random
using Test

Random.seed!(63)

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

@testset verbose = true "HiddenMarkovModels.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(HiddenMarkovModels; ambiguities=false, deps_compat=false)
    end
    @testset verbose = true "Doctests" begin
        doctest(HiddenMarkovModels)
    end
    @testset verbose = true "Discrete markov chain" begin
        include("discrete_markov.jl")
    end
    @testset verbose = true "Hidden Markov Model" begin
        include("hmm.jl")
    end
end
