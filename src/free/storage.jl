"""
    ForwardBackwardStorage{R}

Storage for the forward-backward algorithm applied to a single sequence.

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
    ForwardBackwardLogStorage{R}

Storage for the forward-backward algorithm _in log scale_ applied to a single sequence.

# Fields

- `logα::Matrix{R}`
- `logβ::Matrix{R}`
- `logγ::Matrix{R}`
- `logξ::Array{R,3}`
"""
struct ForwardBackwardLogStorage{R}
    logα::Matrix{R}
    logβ::Matrix{R}
    logγ::Matrix{R}
    logξ::Array{R,3}
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
    MultipleForwardBackwardLogStorage{R}

Storage for the forward-backward algorithm _in log scale_ applied to multiple sequences.

# Fields

- `logα::Vector{Matrix{R}}`
- `logβ::Vector{Matrix{R}}`
- `logγ::Vector{Matrix{R}}`
- `logξ::Vector{Array{R,3}}`
"""
struct MultipleForwardBackwardLogStorage{R}
    logα::Vector{Matrix{R}}
    logβ::Vector{Matrix{R}}
    logγ::Vector{Matrix{R}}
    logξ::Vector{Array{R,3}}
end

"""
    initialize_forward_backward(obs_logdensity)

Create a [`ForwardBackwardStorage`](@ref) based on the number type of the observation log-density matrix.
"""
function initialize_forward_backward(obs_logdensity::AbstractMatrix{L}) where {L<:Real}
    R = float(L)
    S = size(obs_logdensity, 1)
    T = size(obs_logdensity, 2)
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
    initialize_forward_backward_log(obs_logdensity)

Create a [`ForwardBackwardLogStorage`](@ref) based on the number type of the observation log-density matrix.
"""
function initialize_forward_backward_log(obs_logdensity::AbstractMatrix{L}) where {L<:Real}
    S = size(obs_logdensity, 1)
    T = size(obs_logdensity, 2)
    logα = Matrix{L}(undef, S, T)
    logβ = Matrix{L}(undef, S, T)
    logγ = Matrix{L}(undef, S, T)
    logξ = Array{L,3}(undef, S, S, T - 1)
    fb_log_storage = ForwardBackwardLogStorage{L}(logα, logβ, logγ, logξ)
    return fb_log_storage
end

"""
    initialize_forward_backward_multiple_sequences(obs_logdensities)

Create a [`MultipleForwardBackwardStorage`](@ref) based on the number type of the vector of observation density matrices.
"""
function initialize_forward_backward_multiple_sequences(
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
    multiple_fb_storage = MultipleForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return multiple_fb_storage
end

"""
    initialize_forward_backward_log_multiple_sequences(obs_logdensities)

Create a [`MultipleForwardBackwardLogStorage`](@ref) based on the number type of the vector of observation density matrices.
"""
function initialize_forward_backward_log_multiple_sequences(
    obs_logdensities::AbstractVector{<:AbstractMatrix{L}}
) where {L<:Real}
    K = length(obs_logdensities)
    S = size(obs_logdensities[1], 1)
    T = [size(obs_logdensities[k], 2) for k in 1:K]
    logα = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logβ = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logγ = [Matrix{L}(undef, S, T[k]) for k in 1:K]
    logξ = [Array{L,3}(undef, S, S, T[k] - 1) for k in 1:K]
    multiple_fb_log_storage = MultipleForwardBackwardLogStorage{L}(logα, logβ, logγ, logξ)
    return multiple_fb_log_storage
end
