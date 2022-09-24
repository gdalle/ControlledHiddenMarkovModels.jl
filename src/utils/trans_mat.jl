function is_trans_mat(P::AbstractMatrix{R}; atol=1e-5) where {R}
    n, m = size(P)
    n == m || return false
    for i in 1:n
        @views is_prob_vec(P[i, :]; atol=atol) || return false
    end
    return true
end

function uniform_trans_mat(::Type{R}, n::Integer) where {R}
    return ones(R, n, n) ./ n
end

"""
    uniform_trans_mat(n)

Return a transition (stochastic) matrix of size `n` with uniform transition probability distributions.
"""
uniform_trans_mat(n::Integer) = uniform_trans_mat(Float64, n)

"""
    rand_trans_mat(rng, n)

Return a transition (stochastic) matrix of size `n` with random transition probability distributions.
"""
function rand_trans_mat(rng::AbstractRNG, ::Type{R}, n::Integer) where {R}
    P = rand(rng, R, n, n)
    return P ./ sum(P; dims=2)
end

rand_trans_mat(rng::AbstractRNG, n::Integer) = rand_trans_mat(rng, Float64, n)

rand_trans_mat(::Type{R}, n::Integer) where {R} = rand_trans_mat(GLOBAL_RNG, R, n)
rand_trans_mat(n::Integer) = rand_trans_mat(GLOBAL_RNG, n)

"""
    make_trans_mat!(P)

Scale `P` into a transition (stochastic) matrix.
"""
function make_trans_mat!(P::Matrix)
    @views for s in axes(P, 1)
        rowsum = sum(P[s, :])
        P[s, :] .*= inv(rowsum)
    end
    return P
end

function make_trans_mat(P::Matrix)
    rowsums = sum(P, dims=2)
    return P .* inv.(rowsums)
end

"""
    make_log_trans_mat!(logP)

Scale `logP` so that `exp.(logP)` becomes a transition (stochastic) matrix.
"""
function make_log_trans_mat!(logP::Matrix)
    @views for s in axes(logP, 1)
        logP[s, :] .-= logsumexp(logP[s, :])
    end
    return logP
end

function make_log_trans_mat(logP::Matrix)
    rowlogsumexps = [logsumexp(row) for row in eachrow(logP)]
    return logP .- rowlogsumexps
end
