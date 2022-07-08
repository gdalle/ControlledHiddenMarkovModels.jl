"""
    MultivariatePoissonProcess

Multivariate homogeneous temporal Poisson process.

# Fields
- `λ::AbstractVector{<:Real}`: event rates.
"""
struct MultivariatePoissonProcess{R<:Real,V<:AbstractVector{R}} <: AbstractPoissonProcess
    λ::V
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
