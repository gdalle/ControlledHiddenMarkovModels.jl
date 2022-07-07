function baum_welch_multiple_sequences!(
    obs_densities::AbstractVector{<:AbstractMatrix{R}},
    fb_storage::ForwardBackwardStorage{R},
    hmm_init::HMM{R1,R2,Em},
    obs_sequences::AbstractVector;
    max_iterations::Integer=100,
    tol::Real=1e-3,
    show_progress::Bool=true,
) where {R,R1,R2,Em}
    hmm = hmm_init
    S = nb_states(hmm)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, β, γ, ξ, α_sum_inv) = fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Initialize local sufficient statistics for transitions and emissions
    init_count_by_seq = Vector{Vector{R}}(undef, K)
    trans_count_by_seq = Vector{Matrix{R}}(undef, K)
    em_suffstats_by_seq = Matrix{suffstats_type(Em)}(undef, K, S)

    prog = Progress(max_iterations; desc="Baum-Welch algorithm", enabled=show_progress)
    for iteration in 1:max_iterations
        let hmm = hmm
            for k in 1:K
                # Local forward-backward
                update_obs_density!(obs_densities[k], hmm, obs_sequences[k])
                logL_by_seq[k] = forward_backward!(
                    α[k], β[k], γ[k], ξ[k], α_sum_inv[k], hmm, obs_densities[k]
                )
                # Local transition statistics
                init_count_by_seq[k] = γ[k][:, 1]
                trans_count_by_seq[k] = dropdims(sum(ξ[k]; dims=3); dims=3)
                # Local emission statistics
                for i in 1:S
                    x = obs_sequences[k]
                    w = view(γ[k], i, :)  # Distributions.jl doesn't allow this for type of weights, see #1560
                    em_suffstats_by_seq[k, i] = suffstats(Em, x, w)
                end
            end
        end
        push!(logL_evolution, sum(logL_by_seq))

        # Aggregated transitions
        p0 = reduce(+, init_count_by_seq)
        p0 ./= sum(p0)
        P = reduce(+, trans_count_by_seq)
        P ./= sum(P; dims=2)
        # Aggregated emissions
        em = Vector{Em}(undef, S)
        for i in 1:S
            em_suffstats_i = reduce(add_suffstats, view(em_suffstats_by_seq, :, i))
            em[i] = fit_mle(Em, em_suffstats_i)
        end

        hmm = HMM(p0, P, em)

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
    hmm_init::HMM, obs_sequences::AbstractVector, args...; kwargs...
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
function baum_welch(hmm_init::HMM, obs_sequence::AbstractVector, args...; kwargs...)
    return baum_welch_multiple_sequences(hmm_init, [obs_sequence], args...; kwargs...)
end
