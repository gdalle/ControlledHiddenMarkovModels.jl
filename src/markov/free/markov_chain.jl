"""
    MarkovChain{R1,R2}

Discrete-time Markov chain with finite state space.

# Fields
- `p0::Vector{R1}`: initial state distribution.
- `P::Matrix{R2}`: state transition matrix.
"""
struct MarkovChain{R1<:Real,R2<:Real}
    p0::Vector{R1}
    P::Matrix{R2}

    function MarkovChain{R1,R2}(
        p0::AbstractVector{<:Real}, P::AbstractMatrix{<:Real}
    ) where {R1<:Real,R2<:Real}
        @assert is_prob_vec(p0)
        @assert is_trans_mat(P)
        return new{R1,R2}(convert(Vector{R1}, p0), convert(Matrix{R2}, P))
    end
end

function MarkovChain(
    p0::AbstractVector{R1}, P::AbstractMatrix{R2}
) where {R1<:Real,R2<:Real}
    return MarkovChain{R1,R2}(p0, P)
end

@inline DensityInterface.DensityKind(::MarkovChain) = HasDensity()

## Access

nb_states(mc::MarkovChain) = length(mc.p0)
initial_distribution(mc::MarkovChain) = mc.p0
transition_matrix(mc::MarkovChain) = mc.P

"""
    stationary_distribution(mc::MarkovChain)

Compute the equilibrium distribution of a Markov chain using its eigendecomposition.
"""
function stationary_distribution(mc::MarkovChain)
    p_stat = real.(eigvecs(transition_matrix(mc)')[:, end])
    return p_stat / sum(p_stat)
end
