function baum_welch_multiple_sequences!(
    obs_densities::Vector{Matrix{R}},
    fb_storage::ForwardBackwardStorage{R},
    hmm_init::H,
    obs_sequences::AbstractVector;
    max_iterations::Integer=100,
    tol::Real=1e-3,
    show_progress::Bool=true,
) where {R,H<:AbstractHMM}
    hmm = hmm_init
    S = nb_states(hmm)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, β, γ, ξ, α_sum_inv) = fb_storage
    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)
    # Main loop
    prog = Progress(max_iterations; desc="Baum-Welch algorithm", enabled=show_progress)
    for iteration in 1:max_iterations
        for k in 1:K
            # Local forward-backward
            update_obs_density!(obs_densities[k], hmm, obs_sequences[k])
            logL_by_seq[k] = forward_backward!(
                α[k], β[k], γ[k], ξ[k], α_sum_inv[k], hmm, obs_densities[k]
            )
        end
        push!(logL_evolution, sum(logL_by_seq))
        # Aggregated transitions
        p0 = reduce(+, view(γ[k], :, 1) for k in 1:K)
        P = reduce(+, dropdims(sum(ξ[k]; dims=3); dims=3) for k in 1:K)
        p0 ./= sum(p0)
        P ./= sum(P; dims=2)
        # Aggregated emissions
        em = Vector{emission_type(H)}(undef, S)
        xs = (obs_sequences[k] for k in 1:K)
        for i in 1:S
            ws = (view(γ[k], i, :) for k in 1:K)
            em[i] = fit_emission_from_multiple_sequences(H, xs, ws)
        end
        # New object
        hmm = H(p0, P, em)

        if iteration > 1 && (logL_evolution[end] - logL_evolution[end - 1]) / sum(T) < tol
            break
        else
            next!(prog)
        end
    end

    return hmm, logL_evolution
end

"""
    baum_welch_multiple_sequences(hmm_init, obs_sequences)

Run the Baum-Welch algorithm to estimate an [`AbstractHiddenMarkovModel`](@ref) of the same type as `hmm_init`, based on several observation sequences.
"""
function baum_welch_multiple_sequences(
    hmm_init::AbstractHMM, obs_sequences::AbstractVector, args...; kwargs...
)
    K = length(obs_sequences)
    obs_densities = [compute_obs_density(hmm_init, obs_sequences[k]) for k in 1:K]
    fb_storage = initialize_forward_backward_multiple_sequences(obs_densities)
    result = baum_welch_multiple_sequences!(
        obs_densities, fb_storage, hmm_init, obs_sequences, args...; kwargs...
    )
    return result
end

"""
    baum_welch(hmm_init, obs_sequence)

Same as [`baum_welch_multiple_sequences`](@ref) but with a single sequence.
"""
function baum_welch(hmm_init::AbstractHMM, obs_sequence::AbstractVector, args...; kwargs...)
    return baum_welch_multiple_sequences(hmm_init, [obs_sequence], args...; kwargs...)
end
