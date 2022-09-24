
"""
    forward_log!(α, c, obs_logdensity, hmm::AbstractHMM, par)

Perform a forward pass _partly in log scale_ by mutating `α` and `c`.
"""
function forward_log!(
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
    @assert !any(isnan, α)
    return nothing
end

"""
    forward_doublelog!(logα, obs_logdensity, hmm::AbstractHMM, par)

Perform a forward pass _fully in log scale_ by mutating `logα`.
"""
function forward_doublelog!(
    logα::AbstractMatrix, obs_logdensity::AbstractMatrix, hmm::AbstractHMM, par
)
    S, T = size(obs_logdensity)
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, par)
    # Initialization
    @views logα[:, 1] .= logp0 .+ obs_logdensity[:, 1]
    # Recursion
    @views for t in 1:(T - 1)
        for j in 1:S
            logα[j, t + 1] = (
                logsumexp(logα[i, t] + logP[i, j] for i in 1:S) + obs_logdensity[j, t + 1]
            )
        end
    end
    @assert !any(isnan, logα)
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
    return logL
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

    @views logL = logsumexp(logα[:, T])
    return logL
end


"""
    light_forward_log(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass _partly in log scale_ with minimal storage requirements.
"""
function light_forward_log(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
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

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, logL
end

"""
    light_forward_doublelog(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass _fully in log scale_ with minimal storage requirements.
"""
function light_forward_doublelog(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    logp0 = log_initial_distribution(hmm, par)
    logP = log_transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    logα = [logp0[i] + logdensityof(emissions[i], o₁) for i in 1:S]

    # Recursion
    logα_tmp = similar(logα)
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        for j in 1:S
            logα_tmp[j] = logsumexp(logP[i, j] + logα[i] for i in 1:S)
            logα_tmp[j] += logdensityof(emissions[j], oₜ₊₁)
        end
        logα .= logα_tmp
    end

    @assert !any(isnan, logα)
    logL = logsumexp(logα)
    return exp.(logα .- logsumexp(logα)), logL
end
