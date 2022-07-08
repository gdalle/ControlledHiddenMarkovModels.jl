"""
    MultivariatePoissonProcess{R}

Multivariate homogeneous temporal Poisson process.

# Fields
- `λ::Vector{R}`: event rates.
"""
struct MultivariatePoissonProcess{R<:Real} <: AbstractPoissonProcess
    λ::Vector{R}
end

function Base.show(io::IO, pp::MultivariatePoissonProcess{R}) where {R<:Real}
    return print(io, "MultivariatePoissonProcess{$R}($(pp.λ))")
end

## Access

Base.length(pp::MultivariatePoissonProcess) = length(pp.λ)

ground_intensity(pp::MultivariatePoissonProcess) = sum(pp.λ)
log_intensity(pp::MultivariatePoissonProcess, m::Integer) = log(pp.λ[m])

function mark_distribution(pp::MultivariatePoissonProcess)
    return Categorical(pp.λ / sum(pp.λ); check_args=false)
end
