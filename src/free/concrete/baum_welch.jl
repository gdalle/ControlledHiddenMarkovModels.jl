function baum_welch!(
    obs_density::Matrix{R},
    fb_storage::ForwardBackwardStorage{R},
    obs_sequence::AbstractVector,
    hmm_init::H,
    par;
    max_iterations,
    tol,
) where {R,H<:HMM}
    hmm = hmm_init
    S = nb_states(hmm, par)
    T = length(obs_sequence)
    (; α, c, β, bβ, γ, ξ) = fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]

    # Main loop
    @progress for iteration in 1:max_iterations
        # Local forward-backward
        update_obs_density!(obs_density, obs_sequence, hmm, par)
        logL = forward_backward!(α, c, β, bβ, γ, ξ, obs_density, hmm, par)
        push!(logL_evolution, logL)

        # Aggregated transitions
        @views p0 = γ[:, 1]
        P = dropdims(sum(ξ; dims=3); dims=3)
        p0 ./= sum(p0)
        P ./= sum(P; dims=2)

        # Aggregated emissions
        D = emission_type(H)
        emissions = Vector{D}(undef, S)
        @views for s in 1:S
            ws = γ[s, :]
            emissions[s] = fit_mle(D, obs_sequence, ws)
        end

        # New object
        hmm = H(p0, P, emissions)

        if iteration > 1 && (logL_evolution[end] - logL_evolution[end - 1]) / sum(T) < tol
            break
        end
    end

    return hmm, logL_evolution
end

"""
    baum_welch(obs_sequence, hmm_init::HMM[, par; max_iterations, tol])

Apply the Baum-Welch algorithm on a single observation sequence, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modified).
"""
function baum_welch(
    obs_sequence::AbstractVector, hmm_init::HMM, par=nothing; max_iterations=100, tol=1e-3
)
    obs_density = compute_obs_density(obs_sequence, hmm_init, par)
    fb_storage = initialize_forward_backward(obs_density)
    result = baum_welch!(
        obs_density,
        fb_storage,
        obs_sequence,
        hmm_init,
        par;
        max_iterations=max_iterations,
        tol=tol,
    )
    return result
end
