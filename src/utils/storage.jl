struct ForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    β::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
    α_sum_inv::Vector{Vector{R}}
end

function initialize_forward_backward_multiple_sequences(
    obs_densities::AbstractVector{<:AbstractMatrix{R}}
) where {R<:Real}
    K = length(obs_densities)
    S = size(obs_densities[1], 1)
    T = [size(obs_densities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    α_sum_inv = [Vector{R}(undef, T[k]) for k in 1:K]
    fb_storage = ForwardBackwardStorage{R}(α, β, γ, ξ, α_sum_inv)
    return fb_storage
end
