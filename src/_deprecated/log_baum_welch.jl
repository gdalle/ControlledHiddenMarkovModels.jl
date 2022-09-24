"""
    baum_welch_log(obs_sequences, hmm_init::HMM[, par; maxiter, tol])

Apply the Baum-Welch algorithm _partly in log scale_ on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch_log(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    maxiter=100,
    tol=1e-5,
)
    log_od_storage = initialize_obs_logdensities(obs_sequences, hmm_init, par)
    fb_storage = initialize_forward_backward(log_od_storage)
    result = baum_welch_multiple_sequences!(
        log_od_storage, fb_storage, obs_sequences, hmm_init, par; maxiter=maxiter, tol=tol
    )
    return result
end

"""
    baum_welch_doublelog(obs_sequences, hmm_init::HMM[, par; maxiter, tol])

Apply the Baum-Welch algorithm _fully in log scale_ on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch_doublelog(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    maxiter=100,
    tol=1e-5,
)
    log_od_storage = initialize_obs_logdensities(obs_sequences, hmm_init, par)
    log_fb_storage = initialize_forward_backward_log(log_od_storage)
    result = baum_welch_multiple_sequences!(
        log_od_storage,
        log_fb_storage,
        obs_sequences,
        hmm_init,
        par;
        maxiter=maxiter,
        tol=tol,
    )
    return result
end
