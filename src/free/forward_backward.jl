"""
    forward_backward_nolog!(α, c, β, bβ, γ, ξ, obs_density, hmm::AbstractHMM, par)

Apply the full forward-backward algorithm by mutating `α`, `c`, `β`, `bβ`, `γ` and `ξ`.
"""
function forward_backward_nolog!(
    α::AbstractMatrix,
    c::AbstractVector,
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    γ::AbstractMatrix,
    ξ::AbstractArray{<:Any,3},
    obs_density::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_density)
    P = transition_matrix(hmm, par)

    # Forward and backward pass
    forward_nolog!(α, c, obs_density, hmm, par)
    backward_nolog!(β, bβ, c, obs_density, hmm, par)

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

"""
    forward_backward_nolog!(α, c, β, bβ, γ, ξ, obs_logdensity, hmm::AbstractHMM, par)

Apply the full forward-backward algorithm _partly in log scale_ by mutating `α`, `c`, `β`, `bβ`, `γ` and `ξ`.
"""
function forward_backward_log!(
    α::AbstractMatrix,
    c::AbstractVector,
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    γ::AbstractMatrix,
    ξ::AbstractArray{<:Any,3},
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_logdensity)
    P = transition_matrix(hmm, par)

    # Forward and backward pass
    forward_log!(α, c, obs_logdensity, hmm, par)
    backward_log!(β, bβ, c, obs_logdensity, hmm, par)

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

"""
    forward_backward_doublelog!(logα, logβ, logγ, logξ, obs_logdensity, hmm::AbstractHMM, par)

Apply the full forward-backward algorithm _fully in log scale_ by mutating `logα`, `logβ`, `logγ` and `logξ`.
"""
function forward_backward_doublelog!(
    logα::AbstractMatrix,
    logβ::AbstractMatrix,
    logγ::AbstractMatrix,
    logξ::AbstractArray{<:Any,3},
    obs_logdensity::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_logdensity)
    logP = log_transition_matrix(hmm, par)

    # Forward and backward pass
    forward_doublelog!(logα, obs_logdensity, hmm, par)
    backward_doublelog!(logβ, obs_logdensity, hmm, par)

    # State marginals
    logγ .= logα .+ logβ
    @views for t in 1:T
        logγ[:, t] .-= logsumexp(logγ[:, t])
    end

    # Transition marginals
    @views for t in 1:(T - 1)
        for j in 1:S
            for i in 1:S
                logξ[i, j, t] = (
                    logα[i, t] + logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1]
                )
            end
        end
        logξ[:, :, t] .-= logsumexp(logξ[:, :, t])
    end

    logL = logsumexp(logα)
    return float(logL)
end
