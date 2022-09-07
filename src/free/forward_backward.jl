"""
    light_forward(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    obs_logdensity = [logdensityof(emissions[s], o₁) for s in 1:S]
    α = p0 .* exp.(obs_logdensity)
    c = inv(sum(α))
    α .*= c
    logL = -log(c)

    # Recursion
    α_tmp = similar(α)
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        for s in 1:S
            obs_logdensity[s] = logdensityof(emissions[s], oₜ₊₁)
        end
        mul!(α_tmp, P', α)
        α_tmp .*= exp.(obs_logdensity)
        c = inv(sum(α_tmp))
        α_tmp .*= c
        logL -= log(c)
        α .= α_tmp
    end

    @assert !isnan(logL)
    @assert !any(isnan, α)

    return α, float(logL)
end

"""
    forward!(α, c, obs_logdensity, hmm::AbstractHMM, par)

Perform a forward pass by mutating `α` and `c`.
"""
function forward!(
    α::AbstractMatrix,
    c::AbstractVector,
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    _, T = size(obs_logdensity)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)

    # Initialization
    @views α[:, 1] .= p0 .* exp.(obs_logdensity[:, 1])
    @views c[1] = inv(sum(α[:, 1]))
    @views α[:, 1] .*= c[1]

    # Recursion
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], P', α[:, t])
        α[:, t + 1] .*= exp.(obs_logdensity[:, t + 1])
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    # Overflow check
    if @views any(all(iszero_safe, α[:, t]) for t in 1:T)
        throw(OverflowError("Probabilities are too small in forward step."))
    end
    return nothing
end

"""
    backward!(β, bβ, c, obs_logdensity, hmm::AbstractHMM, par)

Perform a backward pass by mutating `β`, `bβ` and `c`.
"""
function backward!(
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    c::AbstractVector,
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    _, T = size(obs_logdensity)
    P = transition_matrix(hmm, par)

    # Initialization
    @views β[:, T] .= one(eltype(β))

    # Recursion
    @views for t in (T - 1):-1:1
        bβ[:, t + 1] .= exp.(obs_logdensity[:, t + 1]) .* β[:, t + 1]
        mul!(β[:, t], P, bβ[:, t + 1])
        β[:, t] .*= c[t]
    end

    # Overflow check
    if @views any(all(iszero_safe, β[:, t]) for t in 1:T)
        throw(OverflowError("Log probabilities are too small in backward step."))
    end
    return nothing
end

"""
    forward_backward!(α, c, β, bβ, γ, ξ, obs_logdensity, hmm::AbstractHMM, par)

Apply the full forward-backward algorithm by mutating `α`, `c`, `β`, `bβ`, `γ` and `ξ`.
"""
function forward_backward!(
    α::AbstractMatrix,
    c::AbstractVector,
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    γ::AbstractMatrix,
    ξ::AbstractArray{<:Real,3},
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_logdensity)
    P = transition_matrix(hmm, par)

    # Forward and backward pass
    forward!(α, c, obs_logdensity, hmm, par)
    backward!(β, bβ, c, obs_logdensity, hmm, par)

    # State marginals
    γ .= α .* β
    @views for t in 1:T
        γ_sum_inv = inv(sum(γ[:, t]))
        γ[:, t] .*= γ_sum_inv
    end

    # Transition marginals
    @views for t in 1:(T - 1)
        for j in 1:S
            for i in 1:S
                ξ[i, j, t] = α[i, t] * P[i, j] * bβ[j, t + 1]
            end
        end
        ξ_sum_inv = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξ_sum_inv
    end

    logL = -sum(log, c)
    return float(logL)
end

function forward_backward_generic!(
    fb_storage::ForwardBackwardStorage,
    k::Integer,
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    (; α, c, β, bβ, γ, ξ) = fb_storage
    return forward_backward!(α[k], c[k], β[k], bβ[k], γ[k], ξ[k], obs_logdensity, hmm, par)
end
