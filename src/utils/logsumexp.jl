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

logsumexp_stream(a::AbstractArray{T}) where {T<:Real} = logsumexp_stream(T, a)

logsumexp_stream(a) = logsumexp_stream(typeof(first(a)), a)

logsumexp(a) = logsumexp_stream(a)
