using Aqua
using ControlledHiddenMarkovModels
using Documenter
using JuliaFormatter
using Random
using Test

Random.seed!(63)

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
    @testset verbose = true "Formatting" begin
        @test format(ControlledHiddenMarkovModels; verbose=false, overwrite=false)
    end
    @testset verbose = true "Doctests" begin
        doctest(ControlledHiddenMarkovModels)
    end
    @testset verbose = true "Plain HMM" begin
        include("plain.jl")
    end
    @testset verbose = true "Parameterized HMM" begin
        include("parameterized.jl")
    end
    @testset verbose = true "Controlled HMM" begin
        include("controlled.jl")
    end
end
