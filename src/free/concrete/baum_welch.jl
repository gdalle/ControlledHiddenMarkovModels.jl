function baum_welch_multiple_sequences!(
    obs_logdensities::Vector{<:Matrix},
    any_fb_storage::AnyForwardBackwardStorage{R},
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

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Main loop
    for iteration in 1:max_iterations
        for k in 1:K
            # Local forward-backward
            update_obs_logdensity!(obs_logdensities[k], obs_sequences[k], hmm, par)
            logL_by_seq[k] = forward_backward_generic!(
                any_fb_storage, k, obs_logdensities[k], hmm, par
            )
        end
        push!(logL_evolution, sum(logL_by_seq))

        p0 = initial_distribution(any_fb_storage)
        P = transition_matrix(any_fb_storage)
        emissions = [
            emission_distribution(H, any_fb_storage, obs_sequences, s) for s in 1:S
        ]

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
    tol=1e-5,
)
    K = length(obs_sequences)
    obs_logdensities = [
        compute_obs_logdensity(obs_sequences[k], hmm_init, par) for k in 1:K
    ]
    fb_storage = initialize_forward_backward(obs_logdensities)
    result = baum_welch_multiple_sequences!(
        obs_logdensities,
        fb_storage,
        obs_sequences,
        hmm_init,
        par;
        max_iterations=max_iterations,
        tol=tol,
    )
    return result
end

"""
    baum_welch_log_multiple_sequences(obs_sequences, hmm_init::HMM[, par; max_iterations, tol])

Apply the Baum-Welch algorithm _in log scale_ on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch_log_multiple_sequences(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    max_iterations=100,
    tol=1e-5,
)
    K = length(obs_sequences)
    obs_logdensities = [
        compute_obs_logdensity(obs_sequences[k], hmm_init, par) for k in 1:K
    ]
    log_fb_storage = initialize_forward_backward_log(obs_logdensities)
    result = baum_welch_multiple_sequences!(
        obs_logdensities,
        log_fb_storage,
        obs_sequences,
        hmm_init,
        par;
        max_iterations=max_iterations,
        tol=tol,
    )
    return result
end

"""
    baum_welch(obs_sequence, hmm_init::HMM[, par; max_iterations, tol])

Apply the Baum-Welch algorithm on a single observation sequence, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modified).
"""
function baum_welch(
    obs_sequence::AbstractVector, hmm_init::HMM, par=nothing; max_iterations=100, tol=1e-5
)
    return baum_welch_multiple_sequences([obs_sequence], hmm_init, par; max_iterations, tol)
end

"""
    baum_welch_log(obs_sequence, hmm_init::HMM[, par; max_iterations, tol])

Apply the Baum-Welch algorithm _in log scale_ on a single observation sequence, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modified).
"""
function baum_welch_log(
    obs_sequence::AbstractVector, hmm_init::HMM, par=nothing; max_iterations=100, tol=1e-5
)
    return baum_welch_log_multiple_sequences(
        [obs_sequence], hmm_init, par; max_iterations, tol
    )
end
