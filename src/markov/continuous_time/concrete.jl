## Structs

"""
    ContinuousMarkovChain

Continuous-time Markov chain with finite state space.

# Fields
- `p0::Vector`: initial state distribution.
- `Q::Matrix`: state rates matrix.
"""
struct ContinuousMarkovChain{R1<:Real,R2<:Real} <: AbstractContinuousMarkovChain
    p0::Vector{R1}
    Q::Matrix{R2}

    function ContinuousMarkovChain{R1,R2}(
        p0::AbstractVector{<:Real}, Q::Matrix{<:Real}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(p0)
        @assert is_rates_mat(Q)
        return new{R1,R2}(convert(Vector{R1}, p0), convert(Matrix{R2}, Q))
    end
end

function ContinuousMarkovChain(
    p0::AbstractVector{R1}, P::AbstractMatrix{R2}
) where {R1<:Real,R2<:Real}
    return ContinuousMarkovChain{R1,R2}(p0, P)
end

## Access

initial_distribution(mc::ContinuousMarkovChain) = mc.p0
intensity_matrix(mc::ContinuousMarkovChain) = mc.Q
