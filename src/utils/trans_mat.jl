function is_trans_mat(P::AbstractMatrix{R}; atol=1e-5) where {R<:Real}
    n, m = size(P)
    n == m || return false
    for i in 1:n
        is_prob_vec(view(P, i, :), atol=atol) || return false
    end
    return true
end

"""
    uniform_trans_mat(n)

Return a stochastic matrix of size `n` with uniform transition probability distributions.
"""
function uniform_trans_mat(::Type{R}, n::Integer) where {R<:Real}
    return ones(R, n, n) ./ n
end

uniform_trans_mat(n::Integer) = uniform_trans_mat(Float64, n)

"""
    rand_trans_mat(n)

Return a stochastic matrix of size `n` with random transition probability distributions.
"""
function rand_trans_mat(::Type{R}, n::Integer) where {R<:Real}
    P = rand(R, n, n)
    return P ./ sum(P; dims=2)
end

rand_trans_mat(n::Integer) = rand_trans_mat(Float64, n)
