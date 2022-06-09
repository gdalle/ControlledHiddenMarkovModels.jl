function Q_transitions(
    hmm::AbstractHMM,
    ξ::AbstractArray{R,3},
    control_sequence::AbstractVector=Fill(nothing, size(ξ, 2)),
    ps=nothing,
    st=nothing,
) where {R}
    S = size(ξ, 1)
    T = size(ξ, 2) + 1
    l = zero(float(R))
    for t in 1:(T - 1)
        uₜ = control_sequence[t]
        Pₜ = transition_matrix(hmm, uₜ, ps, st)
        for j in 1:S, i in 1:S
            l += ξ[i, j, t] * log(Pₜ[i, j])
        end
    end
    return l
end

function Q_emissions(
    hmm::AbstractHMM,
    γ::AbstractMatrix{R},
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector=Fill(nothing, size(ξ, 2)),
    ps=nothing,
    st=nothing,
) where {R}
    S, T = size(γ)
    l = zero(float(R))
    for t in 1:T
        yₜ = obs_sequence[t]
        uₜ = control_sequence[t]
        for i in 1:S
            em_dist = emission_distribution(hmm, i, uₜ, ps, st)
            l += logdensityof(em_dist, yₜ)
        end
    end
    return l
end
