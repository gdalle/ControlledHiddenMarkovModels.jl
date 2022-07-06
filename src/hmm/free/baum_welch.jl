function baum_welch_multiple_sequences!(
    obs_densities::AbstractVector{<:AbstractMatrix{R}},
    fb_storage::ForwardBackwardStorage{R},
    hmm_init::HMM{Tr,Em},
    obs_sequences::AbstractVector,
    control_sequences::AbstractVector=[Fill(nothing, length(seq)) for seq in obs_sequences],
    ps=nothing,
    st=nothing;
    max_iterations::Integer=100,
    tol::Real=1e-3,
    show_progress::Bool=true,
) where {Tr,Em,R}
    hmm = hmm_init
    S = nb_states(hmm)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, β, γ, ξ, α_sum_inv) = fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Initialize local sufficient statistics for transitions and emissions
    transitions_suffstats_by_seq = Vector{suffstats_type(Tr)}(undef, K)
    emissions_suffstats_by_seq = Matrix{suffstats_type(Em)}(undef, K, S)

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
                init_count = γ[k][:, 1]
                trans_count = dropdims(sum(ξ[k]; dims=3); dims=3)
                transitions_suffstats_by_seq[k] = suffstats(Tr, init_count, trans_count)
                # Local emission statistics
                for i in 1:S
                    x = obs_sequences[k]
                    w = view(γ[k], i, :)  # Distributions.jl doesn't allow this for type of weights, see #1560
                    emissions_suffstats_by_seq[k, i] = suffstats(Em, x, w)
                end
            end
        end
        push!(logL_evolution, sum(logL_by_seq))

        # Aggregated transitions
        transitions_suffstats_agg = reduce(add_suffstats, transitions_suffstats_by_seq)
        new_transitions = fit_mle(Tr, transitions_suffstats_agg)
        # Aggregated emissions
        new_emissions = Vector{Em}(undef, S)
        for i in 1:S
            emissions_suffstats_agg_i = reduce(
                add_suffstats, view(emissions_suffstats_by_seq, :, i)
            )
            new_emissions[i] = fit_mle(Em, emissions_suffstats_agg_i)
        end

        hmm = HMM(new_transitions, new_emissions)

        if iteration > 1 && (logL_evolution[end] - logL_evolution[end - 1]) / sum(T) < tol
            break
        else
            next!(prog)
        end
    end

    return hmm, logL_evolution
end
