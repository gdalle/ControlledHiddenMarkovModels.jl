function baum_welch_multiple_sequences!(
    obs_densities::Vector{Matrix{R}},
    fb_storage::ForwardBackwardStorage{R},
    obs_sequences::AbstractVector,
    hmm_init::H,
    par;
    max_iterations::Integer=100,
    tol::Real=1e-3,
) where {R,H<:HMM}
    hmm = hmm_init
    S = nb_states(hmm, par)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, c, β, bβ, γ, ξ) = fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Main loop
    @progress for iteration in 1:max_iterations
        for k in 1:K
            # Local forward-backward
            update_obs_density!(obs_densities[k], obs_sequences[k], hmm, par)
            logL_by_seq[k] = forward_backward!(
                α[k], c[k], β[k], bβ[k], γ[k], ξ[k], obs_densities[k], hmm, par
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
            emissions[s] = fit_from_multiple_sequences(D, xs, ws)
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

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par`.

The parameters are not modified.
"""
function baum_welch_multiple_sequences(
    obs_sequences::AbstractVector, hmm_init::HMM, par=nothing; kwargs...
)
    K = length(obs_sequences)
    obs_densities = [compute_obs_density(obs_sequences[k], hmm_init, par) for k in 1:K]
    fb_storage = initialize_forward_backward_multiple_sequences(obs_densities)
    result = baum_welch_multiple_sequences!(
        obs_densities, fb_storage, obs_sequences, hmm_init, par; kwargs...
    )
    return result
end

"""
    baum_welch(obs_sequence, hmm_init::HMM[, par; max_iterations, tol])

Apply [`baum_welch_multiple_sequences`](@ref) on a single observation sequence.
"""
function baum_welch(obs_sequence::AbstractVector, hmm_init::HMM, par=nothing; kwargs...)
    return baum_welch_multiple_sequences([obs_sequence], hmm_init, par; kwargs...)
end
