function is_rates_mat(Q::AbstractMatrix{R}; atol=1e-5) where {R<:Real}
    n, m = size(Q)
    n == m || return false
    for i in 1:n
        for j in 1:n
            if i != j && Q[i, j] < zero(R)
                return false
            elseif i == j && Q[i, i] > zero(R)
                return false
            end
        end
        if !isapprox(sum(view(Q, i, :)), zero(R); atol=atol)
            return false
        end
    end
    return true
end

"""
    uniform_rates_mat(n)

Return a rates matrix of size `n` with uniform coefficients.
"""
function uniform_rates_mat(::Type{R}, n::Integer) where {R<:Real}
    Q = ones(R, n, n) ./ (n - 1)
    Q[diagind(Q)] .= -one(R)
end

uniform_rates_mat(n::Integer) = uniform_rates_mat(Float64, n)

"""
    rand_rates_mat(rng, n)

Return a stochastic matrix of size `n` with random transition probability distributions.
"""
function rand_rates_mat(rng::AbstractRNG, ::Type{R}, n::Integer) where {R<:Real}
    Q = rand(rng, R, n, n)
    Q[diagind(Q)] .= zero(R)
    Q[diagind(Q)] .= -dropdims(sum(Q; dims=2); dims=2)
    return Q
end

rand_rates_mat(rng::AbstractRNG, n::Integer) = rand_rates_mat(rng, Float64, n)
