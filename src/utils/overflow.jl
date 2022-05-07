# Overflow checks

function all_minus_inf(x::AbstractArray{R}) where {R <: Real}
    for y in x
        y > typemin(R) && return false
    end
    return true
end

function any_minus_inf(x::AbstractArray{R}) where {R <: Real}
    for y in x
        y == typemin(R) && return true
    end
    return false
end

function all_plus_inf(x::AbstractArray{R}) where {R <: Real}
    for y in x
        y < typemax(R) && return false
    end
    return true
end

function any_plus_inf(x::AbstractArray{R}) where {R <: Real}
    for y in x
        y == typemax(R) && return true
    end
    return false
end

function all_zero(x::AbstractArray{R}) where {R <: Real}
    for y in x
        1 / abs(y) < typemax(R) && return false
    end
    return true
end

function any_zero(x::AbstractArray{R}) where {R <: Real}
    for y in x
        1 / abs(y) == typemax(R) && return true
    end
    return false
end
