function DensityInterface.logdensityof(
    mc::AbstractControlledMarkovChain,
    state_sequence::AbstractVector{<:Integer},
    control_sequence::AbstractVector,
    parameters,
)
    T = length(state_sequence)
    logp0 = log_initial_distribution(mc)
    s₁ = state_sequence[1]
    logL = logp0[s₁]
    c₁ = control_sequence[1]
    logP = log_transition_matrix(mc, c₁, parameters)
    for t in 1:(T - 1)
        cₜ = control_sequence[t]
        log_transition_matrix!(logP, mc, cₜ, parameters)
        sₜ, sₜ₊₁ = state_sequence[t], state_sequence[t + 1]
        logL += logP[sₜ, sₜ₊₁]
    end
    return logL
end
