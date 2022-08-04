"""
    LogMultivariatePoissonProcess

Multivariate homogeneous temporal Poisson process.

# Fields
- `logλ::AbstractVector{<:Real}`: log event rates.
"""
struct LogMultivariatePoissonProcess{R<:Real,V<:AbstractVector{R}} <: AbstractPoissonProcess
    logλ::V
end

function Base.show(io::IO, pp::LogMultivariatePoissonProcess{R}) where {R<:Real}
    return print(io, "LogMultivariatePoissonProcess{$R}($(pp.logλ))")
end

## Access

Base.length(pp::LogMultivariatePoissonProcess) = length(pp.logλ)

ground_intensity(pp::LogMultivariatePoissonProcess) = sum(exp, pp.logλ)
log_intensity(pp::LogMultivariatePoissonProcess, m::Integer) = pp.logλ[m]

function mark_distribution(pp::LogMultivariatePoissonProcess)
    return Categorical(exp.(pp.logλ .- logsumexp(pp.logλ)); check_args=false)
end
