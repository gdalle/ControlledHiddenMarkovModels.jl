function baum_welch_multiple_sequences!(
    obs_logdensities::Vector{Matrix{R}},
    multiple_fb_storage::MultipleForwardBackwardStorage{R},
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::H,
    par;
    max_iterations,
    tol,
) where {R,H<:HMM}
    hmm = hmm_init
    S = nb_states(hmm, par)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, c, β, bβ, γ, ξ) = multiple_fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Main loop
    @progress for iteration in 1:max_iterations
        for k in 1:K
            # Local forward-backward
            update_obs_logdensity!(obs_logdensities[k], obs_sequences[k], hmm, par)
            logL_by_seq[k] = forward_backward!(
                α[k], c[k], β[k], bβ[k], γ[k], ξ[k], obs_logdensities[k], hmm, par
            )
        end
        push!(logL_evolution, sum(logL_by_seq))

        # Aggregated transitions
        @views p0 = reduce(+, γ[k][:, 1] for k in 1:K)
        P = reduce(+, dropdims(sum(ξ[k]; dims=3); dims=3) for k in 1:K)
        p0 ./= sum(p0)
        P ./= sum(P; dims=2)

        # Aggregated emissions
        D = emission_type(H)
        emissions = Vector{D}(undef, S)
        xs = (obs_sequences[k] for k in 1:K)
        @views for s in 1:S
            ws = (γ[k][s, :] for k in 1:K)
            emissions[s] = fit_mle_from_multiple_sequences(D, xs, ws)
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
    baum_welch_multiple_sequences(obs_sequences, hmm_init::HMM[, par; max_iterations, tol])

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch_multiple_sequences(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    max_iterations=100,
    tol=1e-3,
)
    K = length(obs_sequences)
    obs_logdensities = [
        compute_obs_logdensity(obs_sequences[k], hmm_init, par) for k in 1:K
    ]
    multiple_fb_storage = initialize_forward_backward_multiple_sequences(obs_logdensities)
    result = baum_welch_multiple_sequences!(
        obs_logdensities,
        multiple_fb_storage,
        obs_sequences,
        hmm_init,
        par;
        max_iterations=max_iterations,
        tol=tol,
    )
    return result
end
