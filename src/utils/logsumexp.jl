function logsumexp(a)
    m = maximum(a)
    se = sum(exp, x - m for x in a)
    return m + log(se)
end

function logsumexp_stream(::Type{T}, a) where {T}
    m = typemin(T)
    se = zero(T)
    for x in a
        if x <= m
            se += exp(x - m)
        else
            se *= exp(m - x)
            se += one(se)
            m = x
        end
    end
    return m + log(se)
end

logsumexp_stream(a::AbstractArray{T}) where {T<:Real} = logsumexp_stream(T, a)

logsumexp_stream(a) = logsumexp_stream(typeof(first(a)), a)
