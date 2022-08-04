"""
    LogMarkedPoissonProcess

Marked homogeneous temporal Poisson process, where each mark is a vector of integers.

# Fields
- `logλ::Real`: log ground intensity
- `logp::AbstractMatrix{<:Real}`: log categorical mark probabilities (dimensions in the columns, possible values in the rows)
"""
struct LogMarkedPoissonProcess{R1<:Real,R2<:Real,M<:AbstractMatrix{R2}} <:
       AbstractPoissonProcess
    logλ::R1
    logp::M
end

function Base.show(io::IO, pp::LogMarkedPoissonProcess{R1,R2}) where {R1<:Real,R2<:Real}
    return print(io, "LogMarkedPoissonProcess{$R1,$R2}($(pp.logλ), $(pp.logp))")
end

## Access

Base.length(pp::LogMarkedPoissonProcess) = size(pp.logp, 2)

ground_intensity(pp::LogMarkedPoissonProcess) = exp(pp.logλ)

function log_intensity(pp::LogMarkedPoissonProcess, m::AbstractVector{<:Integer})
    logI = pp.logλ
    for d in 1:length(pp)
        logI += pp.logp[m[d], d]
    end
    return logI
end

function mark_distribution(pp::LogMarkedPoissonProcess)
    return @views product_distribution([
        Categorical(exp.(pp.logp[:, d] .- logsumexp(pp.logp[:, d])); check_args=false) for
        d in 1:length(pp)
    ])
end
