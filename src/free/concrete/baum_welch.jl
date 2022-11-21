"""
    baum_welch(obs_sequences, hmm_init::HMM[, par; maxiter, tol])

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch(
    obs_sequences, hmm_init::H, par=nothing; maxiter=100, tol=1e-5
) where {H<:HMM}
    ## Initialization
    hmm = hmm_init
    p0 = initial_distribution(hmm, par)
    P = transition_matrix(hmm, par)
    obs_densities = [
        initialize_obs_density(obs_sequence, hmm, par) for obs_sequence in obs_sequences
    ]
    fb_storage = [
        initialize_forward_backward(obs_density, p0, P) for obs_density in obs_densities
    ]
    logL_evolution = Float64[]
    ## EM iterations
    for iteration in 1:maxiter
        logL = 0.0
        for k in eachindex(obs_sequences, obs_densities, fb_storage)
            obs_sequence = obs_sequences[k]
            obs_density = obs_densities[k]
            (; α, c, β, eβ, γ, ξ) = fb_storage[k]
            update_obs_density!(obs_density, obs_sequence, hmm, par)
            logL += forward_backward!(α, c, β, eβ, γ, ξ, obs_density, p0, P)
        end
        push!(logL_evolution, logL)

        p0 = initial_distribution(fb_storage)
        P = transition_matrix(fb_storage)
        emissions = [
            emission_distribution(H, fb_storage, obs_sequences, s) for s in 1:nb_states(hmm)
        ]
        hmm = H(p0, P, emissions)

        if (iteration > 1) && (logL_evolution[end] - logL_evolution[end - 1] < tol)
            break
        end
    end
    return hmm, logL_evolution
end

function initial_distribution(fb_storage)
    @views p0 = Vector(reduce(+, fb_storage[k].γ[:, 1] for k in eachindex(fb_storage)))
    p0 ./= sum(p0)
    @assert !any(isnan, p0)
    return p0
end

function transition_matrix(fb_storage)
    P = reduce(
        +, dropdims(sum(fb_storage[k].ξ; dims=3); dims=3) for k in eachindex(fb_storage)
    )
    P ./= sum(P; dims=2)
    @assert !any(isnan, P)
    return P
end

function emission_distribution(
    ::Type{H}, fb_storage, obs_sequences, s
) where {H<:AbstractHMM}
    D = emission_type(H)
    xs = (obs_sequences[k] for k in eachindex(obs_sequences))
    ws = (fb_storage[k].γ[s, :] for k in eachindex(fb_storage))
    return fit_mle_from_multiple_sequences(D, xs, ws)
end
