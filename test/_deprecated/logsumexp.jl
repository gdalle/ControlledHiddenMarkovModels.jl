function logsumexp(a::AbstractArray{R}) where {R<:Real}
    m = maximum(a)
    lse = m + log(sum(exp, x - m for x in a))
    return lse
end

function logsumexpsum(a::AbstractArray{R}, b::AbstractArray{R}) where {R<:Real}
    m = typemin(R)
    for (xa, xb) in zip(a, b)
        s = xa + xb
        if m < s
            m = s
        end
    end
    lse = m + log(sum(exp, xa + xb - m for (xa, xb) in zip(a, b)))
    return lse
end

function logsumexpsum(
    a::AbstractArray{R}, b::AbstractArray{R}, c::AbstractArray{R}
) where {R<:Real}
    m = typemin(R)
    for (xa, xb, xc) in zip(a, b, c)
        s = xa + xb + xc
        if m < s
            m = s
        end
    end
    lse = m + log(sum(exp, xa + xb + xc - m for (xa, xb, xc) in zip(a, b, c)))
    return lse
end
