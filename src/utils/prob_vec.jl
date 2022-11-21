"""
    is_prob_vec(p; atol)

Check if `p` is a probability distribution vector.
"""
function is_prob_vec(p::AbstractVector{R}; atol=1e-5) where {R}
    return all(>=(zero(R)), p) && isapprox(sum(p), one(R); atol=atol)
end

function uniform_prob_vec(::Type{R}, n::Integer) where {R}
    return ones(R, n) ./ n
end

"""
    uniform_prob_vec(n)

Return a uniform probability distribution vector of size `n`.
"""
uniform_prob_vec(n::Integer) = uniform_prob_vec(Float64, n)

"""
    rand_prob_vec(rng, n)

Return a random probability distribution vector of size `n`.
"""
function rand_prob_vec(rng::AbstractRNG, ::Type{R}, n::Integer) where {R}
    p = rand(rng, R, n)
    return p ./ sum(p)
end

rand_prob_vec(rng::AbstractRNG, n::Integer) = rand_prob_vec(rng, Float64, n)

rand_prob_vec(::Type{R}, n::Integer) where {R} = rand_prob_vec(GLOBAL_RNG, R, n)
rand_prob_vec(n::Integer) = rand_prob_vec(GLOBAL_RNG, n)

"""
    make_prob_vec!(p)

Scale `p` into a probability distribution vector.
"""
function make_prob_vec!(p::AbstractVector)
    p .*= inv(sum(p))
    return p
end

function make_prob_vec(p::AbstractVector)
    return p .* inv(sum(p))
end

"""
    make_log_prob_vec!(logp)

Shift `logp` so that `exp.(logp)` becomes a probability distribution vector.
"""
function make_log_prob_vec!(logp::AbstractVector)
    logp .-= logsumexp(logp)
    return logp
end

function make_log_prob_vec(logp::AbstractVector)
    return logp .- logsumexp(logp)
end
