"""
    MultivariatePoissonProcess{R}

Multivariate homogeneous temporal Poisson process.

# Fields
- `λ::Vector{R}`: event rates.
"""
struct MultivariatePoissonProcess{R<:Real}
    λ::Vector{R}
    mark_dist::Categorical{R,Vector{R}}
end

function MultivariatePoissonProcess(λ::Vector{R}) where {R<:Real}
    mark_dist = Categorical(λ / sum(λ))
    return MultivariatePoissonProcess(λ, mark_dist)
end

function Base.show(io::IO, pp::MultivariatePoissonProcess{R}) where {R<:Real}
    print(io, "MultivariatePoissonProcess{$R}($(pp.λ))")
end

## Prior

Base.@kwdef struct MultivariatePoissonProcessPrior{R1<:Real,R2<:Real}
    λ_α::Vector{R1}
    λ_β::Vector{R2}
end

## Sufficient statistics

Base.@kwdef struct MultivariatePoissonProcessStats{R1<:Real,R2<:Real}
    event_count::Vector{R1}
    duration::R2
end

## Access

Base.length(pp::MultivariatePoissonProcess) = length(pp.λ)

ground_intensity(pp::MultivariatePoissonProcess) = sum(pp.λ)
intensity(pp::MultivariatePoissonProcess, m::Integer) = pp.λ[m]
log_intensity(pp::MultivariatePoissonProcess, m::Integer) = log(pp.λ[m])
mark_distribution(pp::MultivariatePoissonProcess) = pp.mark_dist
