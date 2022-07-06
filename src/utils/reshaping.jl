make_square(x::AbstractVector) = reshape(x, isqrt(length(x)), isqrt(length(x)))

function make_stochastic(x::AbstractMatrix{R}) where {R<:Real}
    y = log1p.(exp.(x))
    return y ./ sum(y; dims=2)
end
