@testset verbose = true "LogSumExp" begin
    a, b, c = rand(10), rand(10), rand(10)
    @test ControlledHiddenMarkovModels.logsumexp(a) ≈ log(sum(exp.(a)))
    @test ControlledHiddenMarkovModels.logsumexpsum(a, b) ≈ log(sum(exp.(a .+ b)))
    @test ControlledHiddenMarkovModels.logsumexpsum(a, b, c) ≈ log(sum(exp.(a .+ b .+ c)))
end
