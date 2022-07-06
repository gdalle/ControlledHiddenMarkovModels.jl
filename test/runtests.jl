using Aqua
using Documenter
using ControlledHiddenMarkovModels
using Random
using Test

Random.seed!(1)

DocMeta.setdocmeta!(
    ControlledHiddenMarkovModels,
    :DocTestSetup,
    :(using ControlledHiddenMarkovModels);
    recursive=true,
)

@testset verbose = true "ControlledHiddenMarkovModels.jl" begin
    @testset verbose = true "Code quality (Aqua.jl)" begin
        Aqua.test_all(ControlledHiddenMarkovModels; ambiguities=false)
    end
    @testset verbose = true "Doctests" begin
        doctest(ControlledHiddenMarkovModels)
    end
    @testset verbose = true "Examples" begin
        @testset verbose = true "Markov chain" begin
            include("examples/markov.jl")
        end
        @testset verbose = true "Controlled Markov chain" begin
            include("examples/markov_controlled.jl")
        end
        @testset verbose = true "Multivariate Poisson process" begin
            include("examples/multivariate_poisson.jl")
        end
        # @testset verbose = true "Hidden Markov Model" begin
        #     include("examples/hmm.jl")
        # end
    end
end
