"""
    DiscreteMarkovChain

Discrete-time Markov chain with finite state space.

# Fields
- `p0::Vector`: initial state distribution.
- `P::Matrix`: state transition matrix.
"""
struct DiscreteMarkovChain{R1<:Real,R2<:Real} <: AbstractDiscreteMarkovChain
    p0::Vector{R1}
    P::Matrix{R2}

    function DiscreteMarkovChain{R1,R2}(
        p0::AbstractVector{<:Real}, P::AbstractMatrix{<:Real}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(p0)
        @assert is_trans_mat(P)
        return new{R1,R2}(convert(Vector{R1}, p0), convert(Matrix{R2}, P))
    end
end

function DiscreteMarkovChain(
    p0::AbstractVector{R1}, P::AbstractMatrix{R2}
) where {R1<:Real,R2<:Real}
    return DiscreteMarkovChain{R1,R2}(p0, P)
end

## Access

initial_distribution(mc::DiscreteMarkovChain) = mc.p0
transition_matrix(mc::DiscreteMarkovChain, args...) = mc.P

function stationary_distribution(mc::DiscreteMarkovChain)
    p_stat = real.(eigvecs(transition_matrix(mc)')[:, end])
    return p_stat / sum(p_stat)
end
