function is_prob_vec(p::AbstractVector{R}; atol=1e-5) where {R<:Real}
    return all(>=(zero(R)), p) && isapprox(sum(p), one(R); atol=atol)
end

"""
    uniform_prob_vec(n)

Return a uniform probability distribution vector of size `n`.
"""
function uniform_prob_vec(::Type{R}, n::Integer) where {R<:Real}
    return ones(R, n) ./ n
end

uniform_prob_vec(n::Integer) = uniform_prob_vec(Float64, n)

"""
    rand_prob_vec(rng, n)

Return a random probability distribution vector of size `n`.
"""
function rand_prob_vec(rng::AbstractRNG, ::Type{R}, n::Integer) where {R<:Real}
    p = rand(rng, R, n)
    return p ./ sum(p)
end

rand_prob_vec(rng::AbstractRNG, n::Integer) = rand_prob_vec(rng, Float64, n)

rand_prob_vec(::Type{R}, n::Integer) where {R} = rand_prob_vec(GLOBAL_RNG, R, n)
rand_prob_vec(n::Integer) = rand_prob_vec(GLOBAL_RNG, n)

make_row_stochastic(x::AbstractArray{<:Real}) = x ./ sum(x; dims=2)
make_column_stochastic(x::AbstractArray{<:Real}) = x ./ sum(x; dims=1)
