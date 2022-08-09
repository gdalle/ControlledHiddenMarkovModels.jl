function light_forward(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]

    # Initialization
    o₁ = obs_sequence[1]
    obs_density = [densityof(emissions[s], o₁) for s in 1:S]
    α = p0 .* obs_density
    c = inv(sum(α))
    α .*= c
    logL = -log(c)

    # Recursion
    α_tmp = similar(α)
    for t in 1:(T - 1)
        oₜ₊₁ = obs_sequence[t + 1]
        for s in 1:S
            obs_density[s] = densityof(emissions[s], oₜ₊₁)
        end
        mul!(α_tmp, P', α)
        α_tmp .*= obs_density
        c = inv(sum(α_tmp))
        α_tmp .*= c
        logL -= log(c)
        α .= α_tmp
    end

    @assert !isnan(logL)
    @assert !any(isnan, α)

    return α, float(logL)
end

function forward!(
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
    # Overflow check
    if @views any(all(iszero_safe, α[:, t]) for t in 1:T)
        throw(OverflowError("Probabilities are too small in forward step."))
    end
    return nothing
end

function backward!(
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

    # Overflow check
    if @views any(all(iszero_safe, β[:, t]) for t in 1:T)
        throw(OverflowError("Log probabilities are too small in backward step."))
    end
    return nothing
end

function forward_backward!(
    α::AbstractMatrix,
    c::AbstractVector,
    β::AbstractMatrix,
    bβ::AbstractMatrix,
    γ::AbstractMatrix,
    ξ::AbstractArray{<:Real,3},
    obs_density::AbstractMatrix,
    hmm::AbstractHMM,
    par,
)
    S, T = size(obs_density)
    P = transition_matrix(hmm, par)

    # Forward and backward pass
    forward!(α, c, obs_density, hmm, par)
    backward!(β, bβ, c, obs_density, hmm, par)

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

function update_obs_density!(
    obs_density::AbstractMatrix, obs_sequence::AbstractVector, hmm::AbstractHMM, par
)
    T, S = length(obs_sequence), nb_states(hmm, par)
    emissions = [emission_distribution(hmm, s, par) for s in 1:S]
    for t in 1:T
        oₜ = obs_sequence[t]
        for s in 1:S
            obs_density[s, t] = densityof(emissions[s], oₜ)
        end
    end
    if @views any(all(iszero_safe, obs_density[:, t]) for t in 1:T)
        throw(OverflowError("Densities are too small for observations."))
    end
    return nothing
end

function compute_obs_density(obs_sequence::AbstractVector, hmm::AbstractHMM, par)
    T, S = length(obs_sequence), nb_states(hmm, par)
    test_density_value = densityof(emission_distribution(hmm, 1, par), obs_sequence[1])
    obs_density = Matrix{typeof(test_density_value)}(undef, S, T)
    update_obs_density!(obs_density, obs_sequence, hmm, par)
    return obs_density
end

struct ForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    c::Vector{Vector{R}}
    β::Vector{Matrix{R}}
    bβ::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
end

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
    fb_storage = ForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return fb_storage
end
