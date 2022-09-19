function baum_welch_multiple_sequences!(
    od_storage::AnyObsDensityStorage,
    fb_storage::AnyForwardBackwardStorage,
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::H,
    par;
    maxiter,
    tol,
) where {H<:HMM}
    hmm = hmm_init
    R = get_logtype(od_storage)
    logL_evolution = R[]
    for iteration in 1:maxiter
        update_obs_densities_generic!(od_storage, obs_sequences, hmm, par)
        logL = forward_backward_generic!(fb_storage, od_storage, hmm, par)
        push!(logL_evolution, logL)

        p0 = initial_distribution(fb_storage)
        P = transition_matrix(fb_storage)
        @assert !any(isnan, P)
        emissions = [
            emission_distribution(H, fb_storage, obs_sequences, s) for s in 1:nb_states(hmm)
        ]
        hmm = H(p0, P, emissions)

        if (iteration > 1) && (logL_evolution[end] - logL_evolution[end - 1] < tol)
            break
        end
    end
    update_obs_densities_generic!(od_storage, obs_sequences, hmm, par)
    logL = forward_backward_generic!(fb_storage, od_storage, hmm, par)
    push!(logL_evolution, logL)
    return hmm, logL_evolution
end

"""
    baum_welch_nolog(obs_sequences, hmm_init::HMM[, par; maxiter, tol])

Apply the Baum-Welch algorithm on multiple observation sequences, starting from an initial [`HMM`](@ref) `hmm_init` with parameters `par` (not modifed).
"""
function baum_welch_nolog(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    maxiter=100,
    tol=1e-5,
)
    od_storage = initialize_obs_densities(obs_sequences, hmm_init, par)
    fb_storage = initialize_forward_backward(od_storage)
    result = baum_welch_multiple_sequences!(
        od_storage, fb_storage, obs_sequences, hmm_init, par; maxiter=maxiter, tol=tol
    )
    return result
end

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

function baum_welch(
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm_init::HMM,
    par=nothing;
    maxiter=100,
    tol=1e-5,
    safe=2,
)
    if safe == 0
        return baum_welch_nolog(obs_sequences, hmm_init, par; maxiter=maxiter, tol=tol)
    elseif safe == 1
        return baum_welch_log(obs_sequences, hmm_init, par; maxiter=maxiter, tol=tol)
    elseif safe == 2
        return baum_welch_doublelog(obs_sequences, hmm_init, par; maxiter=maxiter, tol=tol)
    end
end
