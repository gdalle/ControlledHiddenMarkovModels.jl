function Q_function(
    hmm::AbstractHMM,
    γ::AbstractMatrix{R},
    ξ::AbstractArray{R,3},
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector=Fill(nothing, size(γ, 1)),
    ps=nothing,
    st=nothing,
) where {R}
    S, T = size(γ)
    l = zero(float(R))
    for t in 1:T
        yₜ = obs_sequence[t]
        uₜ = control_sequence[t]
        Pₜ, emissionsₜ = transition_matrix_and_emission_distributions(hmm, uₜ, ps, st)
        for i in 1:S
            l += γ[i, t] * logdensityof(emissionsₜ[i], yₜ)
        end
        if t < T
            for j in 1:S, i in 1:S
                l += ξ[i, j, t] * log(Pₜ[i, j])
            end
        end
    end
    return l
end
