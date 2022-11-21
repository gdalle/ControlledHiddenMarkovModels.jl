"""
    forward!(α, c, obs_density, p0, P)

Perform a forward pass by mutating `α` and `c`.
"""
function forward!(α::Matrix, c::Vector, obs_density::Matrix, p0::Vector, P::Matrix)
    T = size(α, 2)
    @views α[:, 1] .= p0 .* obs_density[:, 1]
    @views c[1] = inv(sum(α[:, 1]))
    @views α[:, 1] .*= c[1]
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], P', α[:, t])
        α[:, t + 1] .*= obs_density[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    @assert !any(isnan, α)
    return nothing
end

"""
    backward!(β, eβ, c, obs_density, P)

Perform a backward pass by mutating `β` and `eβ`. Comes after forward pass.
"""
function backward!(β::Matrix, eβ::Matrix, c::Vector, obs_density::Matrix, P::Matrix)
    T = size(β, 2)
    @views β[:, T] .= one(eltype(β))
    @views for t in (T - 1):-1:1
        eβ[:, t + 1] .= obs_density[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], P, eβ[:, t + 1])
        β[:, t] .*= c[t]
    end
    @assert !any(isnan, β)
    return nothing
end

"""
    marginals!(γ, ξ, α, β, eβ, P)

Compute state and transition marginals by mutating `γ` and `ξ`. Comes after backward pass.
"""
function marginals!(γ::Matrix, ξ::Array, α::Matrix, β::Matrix, eβ::Matrix, P::Matrix)
    S, T = size(γ)
    γ .= α .* β
    @views for t in 1:T
        γ_sum_inv = inv(sum(γ[:, t]))
        γ[:, t] .*= γ_sum_inv
    end
    @assert !any(isnan, γ)
    @views for t in 1:(T - 1)
        for j in 1:S
            for i in 1:S
                ξ[i, j, t] = α[i, t] * P[i, j] * eβ[j, t + 1]
            end
        end
        ξ_sum_inv = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξ_sum_inv
    end
    @assert !any(isnan, ξ)
    return nothing
end

"""
    forward_backward!(α, c, β, eβ, γ, ξ, obs_density, p0, P)

Apply the full forward-backward algorithm by mutating `α`, `c`, `β`, `eβ`, `γ` and `ξ`.
"""
function forward_backward!(
    α::Matrix,
    c::Vector,
    β::Matrix,
    eβ::Matrix,
    γ::Matrix,
    ξ::Array{<:Any,3},
    obs_density::Matrix,
    p0::Vector,
    P::Matrix,
)
    forward!(α, c, obs_density, p0, P)
    backward!(β, eβ, c, obs_density, P)
    marginals!(γ, ξ, α, β, eβ, P)
    logL = -sum(log, c)
    return logL
end

function initialize_forward_backward(obs_density::Matrix, p0::Vector, P::Matrix)
    S, T = size(obs_density)
    R = promote_type(eltype(p0), eltype(P), eltype(obs_density))
    α = Matrix{R}(undef, S, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, S, T)
    eβ = Matrix{R}(undef, S, T)
    γ = Matrix{R}(undef, S, T)
    ξ = Array{R,3}(undef, S, S, T - 1)
    return (α=α, c=c, β=β, eβ=eβ, γ=γ, ξ=ξ)
end

function forward_backward(obs_density::Matrix, hmm::AbstractHMM, par)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    (; α, c, β, eβ, γ, ξ) = initialize_forward_backward(obs_density, p0, P)
    forward_backward!(α, c, β, eβ, γ, ξ, obs_density, p0, P)
    return (α=α, c=c, β=β, eβ=eβ, γ=γ, ξ=ξ)
end
