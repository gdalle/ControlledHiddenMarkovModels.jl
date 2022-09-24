"""
    light_forward(obs_sequence, hmm::AbstractHMM, par)

Perform a lightweight forward pass with minimal storage requirements.
"""
function light_forward(obs_sequence, hmm::AbstractHMM, par)
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

    @assert !any(isnan, α)
    @assert !isnan(logL)
    return α, logL
end
