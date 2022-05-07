function is_prob_vec(p::AbstractVector{R}) where {R<:Real}
    return all(>=(zero(R)), p) && sum(p) ≈ one(R)
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
    rand_prob_vec(n)

Return a random probability distribution vector of size `n`.
"""
function rand_prob_vec(::Type{R}, n::Integer) where {R<:Real}
    p = rand(R, n)
    return p ./ sum(p)
end

rand_prob_vec(n::Integer) = rand_prob_vec(Float64, n)
