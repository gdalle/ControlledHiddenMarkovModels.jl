"""
    logsumexp_stream(::Type{T}, a)

Compute the logsumexp function in a single pass for an iterable `a` with elements of type `T`.
Source: <https://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html>
"""
function logsumexp_stream(::Type{T}, a) where {T}
    m = typemin(T)
    se = zero(T)
    for x in a
        if x < m
            se += exp(x - m)
        elseif x == m
            se += one(se)
        else
            se *= exp(m - x)
            se += one(se)
            m = x
        end
    end
    lse = m + log(se)
    @assert !isnan(lse)
    return lse
end

logsumexp_stream(a::AbstractArray{T}) where {T} = logsumexp_stream(T, a)

logsumexp_stream(a) = logsumexp_stream(typeof(first(a)), a)

"""
    logsumexp(a)

Use [`logsumexp_stream`](@ref) to compute the logsumexp of an iterable `a`.
"""
logsumexp(a) = logsumexp_stream(a)
