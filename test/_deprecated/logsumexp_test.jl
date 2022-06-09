@testset verbose = true "LogSumExp" begin
    a, b, c = rand(10), rand(10), rand(10)
    @test HiddenMarkovModels.logsumexp(a) ≈ log(sum(exp.(a)))
    @test HiddenMarkovModels.logsumexpsum(a, b) ≈ log(sum(exp.(a .+ b)))
    @test HiddenMarkovModels.logsumexpsum(a, b, c) ≈ log(sum(exp.(a .+ b .+ c)))
end
