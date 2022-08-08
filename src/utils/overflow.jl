function iszero_safe(x::R) where {R<:Real}
    return inv(abs(x)) == typemax(R)
end
