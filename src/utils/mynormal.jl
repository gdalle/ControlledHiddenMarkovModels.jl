function add_suffstats(stats1::Distributions.NormalStats, stats2::Distributions.NormalStats)
    s = stats1.s + stats2.s
    s2 = stats1.s2 + stats2.s2
    tw = stats1.tw + stats2.tw
    m = s / tw
    stats = Distributions.NormalStats(s, m, s2, tw)
    return stats
end

function fit_mle_from_multiple_sequences(D::Type{<:Distributions.Normal}, xs, ws)
    stats = reduce(add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws))
    return fit_mle(D, stats)
end

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
