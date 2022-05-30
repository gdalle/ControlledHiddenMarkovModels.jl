function normalizing_weight(x::AbstractArray{<:Real})
    s = sum(x)
    weight(t) = x[t] / s
    return weight
end

# Normal

function get_onlinestat_type(::Type{D}) where {T,D<:Normal{T}}
    return FitNormal{<:Variance{T,T}}
end

function get_onlinestat(::Type{D}; weight) where {T,D<:Normal{T}}
    return FitNormal(Variance(T; weight=weight))
end

function Distributions.fit_mle(::Type{D}, os::FitNormal) where {D<:Normal}
    return D(mean(os), std(os))
end

# Poisson process
