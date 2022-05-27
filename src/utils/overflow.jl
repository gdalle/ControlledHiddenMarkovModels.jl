function iszero_safe(x::R) where {R<:Real}
    return 1 / abs(x) == typemax(R)
end
