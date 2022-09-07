"""
    HiddenMarkovModel{R1,R2,D}

Concrete subtype of [`AbstractHMM`](@ref) which stores the state and observation parameters directly.

# Fields

- `p0::Vector{R1}`
- `P::Matrix{R2}`
- `emissions::Vector{D}`

# Compatible with

- - [`baum_welch_multiple_sequences(obs_sequences, hmm_init, par; kwargs...)`](@ref)
- - [`baum_welch(obs_sequence, hmm_init, par; kwargs...)`](@ref)
"""
struct HiddenMarkovModel{R1,R2,D} <: AbstractHMM
    p0::Vector{R1}
    P::Matrix{R2}
    emissions::Vector{D}
end

"""
    HMM

Alias for [`HiddenMarkovModel`](@ref).
"""
const HMM = HiddenMarkovModel

nb_states(hmm::HMM, par=nothing) = length(hmm.p0)
initial_distribution(hmm::HMM, par=nothing) = hmm.p0
transition_matrix(hmm::HMM, par=nothing) = hmm.P
emission_distribution(hmm::HMM, s::Integer, par=nothing) = hmm.emissions[s]

"""
    emission_type(::Type{<:HMM})

Return the type of an emission distribution object.
"""
emission_type(::Type{HMM{R1,R2,D}}) where {R1,R2,D} = D

"""
    fit_mle_from_single_sequence(::Type{D}, x, w)

Fit a distribution of type `D` based on a single sequence of observations `x` associated with a single sequence of weights `w`.

Defaults to `Distributions.fit_mle`, with a special case for vectors of vectors (because `Distributions.fit_mle` accepts matrices instead).
Users are free to override this default for concrete distributions.
"""
function fit_mle_from_single_sequence(
    T::Type{D}, x::AbstractVector, w::AbstractVector
) where {D}
    return fit_mle(T, x, w)
end

function fit_mle_from_single_sequence(
    T::Type{D}, x::AbstractVector{<:AbstractVector}, w::AbstractVector
) where {D}
    return fit_mle(T, hcat(x...), w)
end

"""
    fit_mle_from_multiple_sequences(::Type{D}, xs, ws)

Fit a distribution of type `D` based on multiple sequences of observations `xs` associated with multiple sequences of weights `ws`.

Must accept arbitrary iterables for `xs` and `ws`.
"""
function fit_mle_from_multiple_sequences(::Type{D}, xs, ws) where {D}
    return error(
        "The method `fit_mle_from_multiple_sequences(::Type{D}, xs, ws)` is not implemented for emission type $D. It is required for the Baum-Welch algorithm when applied to multiple sequences.",
    )
end
