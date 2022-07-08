"""
    MarkedPoissonProcess{R1,R2}

Marked homogeneous temporal Poisson process, where each mark is a vector of integers.

# Fields
- `λ::R1`: ground intensity
- `mark_probs::Matrix{R2}`: categorical mark probabilities (dimensions in the columns, possible values in the rows)
"""
struct MarkedPoissonProcess{R1<:Real,R2<:Real} <: AbstractPoissonProcess
    λ::R1
    mark_probs::Matrix{R2}
end

function Base.show(io::IO, pp::MarkedPoissonProcess{R1,R2}) where {R1<:Real,R2<:Real}
    return print(io, "MarkedPoissonProcess{$R1,$R2}($(pp.λ), $(pp.mark_probs))")
end

## Access

Base.length(pp::MarkedPoissonProcess) = length(pp.proddist)

ground_intensity(pp::MarkedPoissonProcess) = pp.λ

function log_intensity(pp::MarkedPoissonProcess, m::AbstractVector{<:Integer})
    return log(pp.λ) + sum(log, pp.mark_probs[m[d], d] for d in 1:length(pp))
end

function mark_distribution(pp::MarkedPoissonProcess)
    return @views product_distribution([
        Categorical(pp.mark_probs[:, d]; check_args=false) for d in 1:size(pp.mark_probs, 2)
    ])
end
