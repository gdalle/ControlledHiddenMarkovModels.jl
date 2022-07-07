function DensityInterface.logdensityof(
    mc::AbstractControlledMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractMatrix{<:Real},
    args...,
)
    p0 = initial_distribution(mc)
    P_all = transition_matrix(mc, control_sequence, args...)
    T = length(state_sequence)
    i₁ = state_sequence[1]
    l = log(p0[i₁])
    for t in 1:(T - 1)
        iₜ, iₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        Pₜ = view(P_all, :, :, t)
        l += log(Pₜ[iₜ, iₜ₊₁])
    end
    return l
end
