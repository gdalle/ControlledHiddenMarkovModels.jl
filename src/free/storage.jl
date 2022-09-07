"""
    ForwardBackwardStorage{R}

Storage for the forward-backward algorithm applied a single sequence.

# Fields

- `α::Matrix{R}`
- `c::Vector{R}`
- `β::Matrix{R}`
- `bβ::Matrix{R}`
- `γ::Matrix{R}`
- `ξ::Array{R,3}`
"""
struct ForwardBackwardStorage{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    bβ::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
end

"""
    MultipleForwardBackwardStorage{R}

Storage for the forward-backward algorithm applied to multiple sequences.

# Fields

- `α::Vector{Matrix{R}}`
- `c::Vector{Vector{R}}`
- `β::Vector{Matrix{R}}`
- `bβ::Vector{Matrix{R}}`
- `γ::Vector{Matrix{R}}`
- `ξ::Vector{Array{R,3}}`
"""
struct MultipleForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    c::Vector{Vector{R}}
    β::Vector{Matrix{R}}
    bβ::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
end

"""
    initialize_forward_backward(obs_density)

Create a [`ForwardBackwardStorage`](@ref) with same number type as the observation density matrix.
"""
function initialize_forward_backward(obs_density::AbstractMatrix{R}) where {R<:Real}
    S = size(obs_density, 1)
    T = size(obs_density, 2)
    α = Matrix{R}(undef, S, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, S, T)
    bβ = Matrix{R}(undef, S, T)
    γ = Matrix{R}(undef, S, T)
    ξ = Array{R,3}(undef, S, S, T - 1)
    fb_storage = ForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return fb_storage
end

"""
    initialize_forward_backward_multiple_sequences(obs_densities)

Create a [`MultipleForwardBackwardStorage`](@ref) with same number type as the vector of observation density matrices.
"""
function initialize_forward_backward_multiple_sequences(
    obs_densities::AbstractVector{<:AbstractMatrix{R}}
) where {R<:Real}
    K = length(obs_densities)
    S = size(obs_densities[1], 1)
    T = [size(obs_densities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    c = [Vector{R}(undef, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    bβ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    multiple_fb_storage = MultipleForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return multiple_fb_storage
end
