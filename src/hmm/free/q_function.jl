function Q_function(
    hmm::AbstractHMM,
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    obs_sequence::AbstractVector,
) where {R}
    S, T = size(γ)
    p0 = initial_distribution(hmm)
    P = transition_matrix(hmm)
    em = emissions(hmm)
    l = zero(float(R))
    for i in 1:S
        l += γ[i, 1] * log(p0[i])
    end
    for t in 1:(T - 1)
        for j in 1:S, i in 1:S
            l += ξ[i, j, t] * log(P[i, j])
        end
    end
    for t in 1:T
        yₜ = obs_sequence[t]
        for i in 1:S
            l += γ[i, t] * logdensityof(em[i], yₜ)
        end
    end
    return l
end
