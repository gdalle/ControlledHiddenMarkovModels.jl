"""
    forward_nolog!(α, c, obs_density, hmm::AbstractHMM, par)

Perform a forward pass by mutating `α` and `c`.
"""
function forward_nolog!(
    α::AbstractMatrix, c::AbstractVector, obs_density::AbstractMatrix, hmm::AbstractHMM, par
)
    _, T = size(obs_density)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    # Initialization
    @views α[:, 1] .= p0 .* obs_density[:, 1]
    @views c[1] = inv(sum(α[:, 1]))
    @views α[:, 1] .*= c[1]
    # Recursion
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
