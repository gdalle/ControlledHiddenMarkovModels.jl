"""
    iszero_safe(x::R)

Check if a number `x` is zero by comparing its inverse with `typemax(R)`.

This is useful in the following case:
```julia
julia> x = 1e-320
1.0e-320

julia> iszero(x)
false

julia> inv(x)
Inf
```
"""
function iszero_safe(x::R) where {R<:Real}
    return inv(abs(x)) >= typemax(R)
end
