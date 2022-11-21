struct MyNormal{R1,R2}
    μ::R1
    σ::R2
end

@inline DensityInterface.DensityKind(::MyNormal) = HasDensity()

function Base.rand(rng::AbstractRNG, normal::MyNormal{R1,R2}) where {R1,R2}
    R = promote_type(R1, R2)
    return normal.μ + normal.σ * randn(rng, R)
end

function Base.rand(normal::MyNormal)
    return rand(GLOBAL_RNG, normal)
end

function DensityInterface.logdensityof(normal::MyNormal, x::Number)
    return -log(normal.σ) - (x - normal.μ)^2 / (2 * normal.σ^2)
end

function Distributions.fit_mle(
    ::Type{MyNormal{R1,R2}}, x::AbstractVector, w::AbstractVector
) where {R1,R2}
    μ = sum(xᵢ * wᵢ for (xᵢ, wᵢ) in zip(x, w)) / sum(w)
    σ = sqrt(sum((xᵢ - μ)^2 * wᵢ for (xᵢ, wᵢ) in zip(x, w)) / sum(w))
    return MyNormal(R1(μ), R2(σ))
end
