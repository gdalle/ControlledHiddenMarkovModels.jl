"""
    MarkedPoissonProcess

Marked homogeneous temporal Poisson process, where each mark is a vector of integers.

# Fields
- `λ::Real`: ground intensity
- `p::AbstractMatrix{<:Real}`: categorical mark probabilities (dimensions in the columns, possible values in the rows)
"""
struct MarkedPoissonProcess{R1<:Real,R2<:Real,M<:AbstractMatrix{R2}} <:
       AbstractPoissonProcess
    λ::R1
    p::M
end

function Base.show(io::IO, pp::MarkedPoissonProcess{R1,R2}) where {R1<:Real,R2<:Real}
    return print(io, "MarkedPoissonProcess{$R1,$R2}($(pp.λ), $(pp.p))")
end

## Access

Base.length(pp::MarkedPoissonProcess) = size(pp.p, 2)

ground_intensity(pp::MarkedPoissonProcess) = pp.λ

function log_intensity(pp::MarkedPoissonProcess, m::AbstractVector{<:Integer})
    logI = log(pp.λ)
    for d in 1:length(pp)
        logI += log(pp.p[m[d], d])
    end
    return logI
end

function mark_distribution(pp::MarkedPoissonProcess)
    return @views product_distribution([
        Categorical(pp.p[:, d]; check_args=false) for d in 1:length(pp)
    ])
end
