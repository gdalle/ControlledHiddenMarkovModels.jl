"""
    backward_nolog!(β, bβ, c, obs_density, hmm::AbstractHMM, par)

Perform a backward pass by mutating `β`, `bβ` and `c`.
"""
function backward_nolog!(
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    c::AbstractVector,
    obs_density::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    _, T = size(obs_density)
    P = transition_matrix(hmm, par)
    # Initialization
    @views β[:, T] .= one(eltype(β))
    # Recursion
    @views for t in (T - 1):-1:1
        bβ[:, t + 1] .= obs_density[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], P, bβ[:, t + 1])
        β[:, t] .*= c[t]
    end
    @assert !any(isnan, β)
    return nothing
end

"""
    backward_log!(β, bβ, c, obs_logdensity, hmm::AbstractHMM, par)

Perform a backward pass by mutating `β`, `bβ` and `c`.
"""
function backward_log!(
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
    @assert !any(isnan, β)
    return nothing
end

"""
    backward_doublelog!(logβ, obs_logdensity, hmm::AbstractHMM, par)

Perform a backward pass _fully in log scale_ by mutating `logβ`.
"""
function backward_doublelog!(
    logβ::AbstractMatrix, obs_logdensity::AbstractMatrix, hmm::AbstractHMM, par
)
    S, T = size(obs_logdensity)
    logP = log_transition_matrix(hmm, par)
    @assert !any(isnan, logP)
    # Initialization
    @views logβ[:, T] .= zero(eltype(logβ))
    # Recursion
    @views for t in (T - 1):-1:1
        for i in 1:S
            logβ[i, t] = logsumexp(
                logP[i, j] + obs_logdensity[j, t + 1] + logβ[j, t + 1] for j in 1:S
            )
        end
    end
    @assert !any(isnan, logβ)
    return nothing
end
