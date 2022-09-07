"""
    ForwardBackwardStorage{R}

Storage for the forward-backward algorithm applied to multiple sequences.

# Fields

- `α::Vector{Matrix{R}}`
- `c::Vector{Vector{R}}`
- `β::Vector{Matrix{R}}`
- `bβ::Vector{Matrix{R}}`
- `γ::Vector{Matrix{R}}`
- `ξ::Vector{Array{R,3}}`
"""
struct ForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    c::Vector{Vector{R}}
    β::Vector{Matrix{R}}
    bβ::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
end

"""
    initialize_forward_backward(obs_logdensities)

Create a [`ForwardBackwardStorage`](@ref) based on the number type of the vector of observation density matrices.
"""
function initialize_forward_backward(
    obs_logdensities::AbstractVector{<:AbstractMatrix{L}}
) where {L<:Real}
    R = float(L)
    K = length(obs_logdensities)
    S = size(obs_logdensities[1], 1)
    T = [size(obs_logdensities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    c = [Vector{R}(undef, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    bβ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    multiple_fb_storage = ForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return multiple_fb_storage
end

"""
    LogForwardBackwardStorage{R}

Storage for the forward-backward algorithm _in log scale_ applied to multiple sequences.

# Fields

- `logα::Vector{Matrix{R}}`
- `logβ::Vector{Matrix{R}}`
- `logγ::Vector{Matrix{R}}`
- `logξ::Vector{Array{R,3}}`
"""
struct LogForwardBackwardStorage{R}
    logα::Vector{Matrix{R}}
    logβ::Vector{Matrix{R}}
    logγ::Vector{Matrix{R}}
    logξ::Vector{Array{R,3}}
end

"""
    initialize_forward_backward_log(obs_logdensities)

Create a [`LogForwardBackwardStorage`](@ref) based on the number type of the vector of observation density matrices.
"""
function initialize_forward_backward_log(
    obs_logdensities::AbstractVector{<:AbstractMatrix{L}}
) where {L<:Real}
    K = length(obs_logdensities)
    S = size(obs_logdensities[1], 1)
    T = [size(obs_logdensities[k], 2) for k in 1:K]
    logα = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logβ = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logγ = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logξ = [Array{L,3}(undef, S, S, T[k] - 1) for k in 1:K]
    multiple_log_fb_storage = LogForwardBackwardStorage{L}(logα, logβ, logγ, logξ)
    return multiple_log_fb_storage
end

const AnyForwardBackwardStorage{R} = Union{
    ForwardBackwardStorage{R},LogForwardBackwardStorage{R}
}
