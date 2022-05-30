function initialize_baum_welch_multiple_sequences(
    obs_densities::AbstractVector{<:AbstractMatrix{R}}
) where {R<:Real}
    K = length(obs_densities)
    S = size(obs_densities[1], 1)
    T = [size(obs_densities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    α_sum_inv = [Vector{R}(undef, T[k]) for k in 1:K]
    tup = (α=α, β=β, γ=γ, ξ=ξ, α_sum_inv=α_sum_inv)
    return tup
end

function baum_welch_multiple_sequences!(
    obs_densities::AbstractVector{<:AbstractMatrix{R}},
    α::AbstractVector{<:AbstractMatrix{R}},
    β::AbstractVector{<:AbstractMatrix{R}},
    γ::AbstractVector{<:AbstractMatrix{R}},
    ξ::AbstractVector{<:AbstractArray{R,3}},
    α_sum_inv::AbstractVector{<:AbstractVector{R}},
    hmm_init::HMM{Tr,Em},
    obs_sequences::AbstractVector;
    max_iterations::Integer=100,
    tol::Real=1e-3,
    show_progress::Bool=true,
) where {Tr,Em,R}
    hmm = hmm_init
    S = nb_states(hmm)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    # Initialize local sufficient statistics for transitions and emissions
    init_count_by_seq = Vector{Vector{R}}(undef, K)
    trans_count_by_seq = Vector{Matrix{R}}(undef, K)
    emissions_stats_by_seq = Matrix{get_onlinestat_type(Em)}(undef, K, S)

    prog = Progress(max_iterations; desc="Baum-Welch algorithm", enabled=show_progress)
    for iteration in 1:max_iterations
        let hmm = hmm
            @threads for k in 1:K
                # Local forward-backward
                update_obs_density!(obs_densities[k], hmm, obs_sequences[k])
                logL_by_seq[k] = forward_backward!(
                    α[k], β[k], γ[k], ξ[k], α_sum_inv[k], hmm, obs_densities[k]
                )
                # Local transition statistics
                init_count_by_seq[k] = γ[k][:, 1]
                trans_count_by_seq[k] = dropdims(sum(ξ[k]; dims=3); dims=3)
                # Local emission statistics
                for s in 1:S
                    emstat = get_onlinestat(Em; weight=normalizing_weight(view(γ[k], s, :)))
                    fit!(emstat, obs_sequences[k])
                    emissions_stats_by_seq[k, s] = emstat
                end
            end
        end
        push!(logL_evolution, sum(logL_by_seq))

        # Aggregated transitions
        init_count_agg = ThreadsX.sum(init_count_by_seq)
        trans_count_agg = ThreadsX.sum(trans_count_by_seq)
        new_transitions = fit_mle(Tr, init_count_agg, trans_count_agg)
        # Aggregated emissions
        new_emissions = Vector{Em}(undef, S)
        @threads for s in 1:S
            emissions_stats_s = emissions_stats_by_seq[1, s]
            for k in 2:K
                merge!(emissions_stats_s, emissions_stats_by_seq[k, s])
            end
            new_emissions[s] = fit_mle(Em, emissions_stats_s)
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

"""
    baum_welch_multiple_sequences(hmm_init, obs_sequences)

Run the Baum-Welch algorithm to estimate a [`HiddenMarkovModel`](@ref) of the same type as `hmm_init`, based on several observation sequences.
"""
function baum_welch_multiple_sequences(
    hmm_init::HMM, obs_sequences::AbstractVector; kwargs...
)
    K = length(obs_sequences)
    obs_densities = [compute_obs_density(hmm_init, obs_sequences[k]) for k in 1:K]
    (; α, β, γ, ξ, α_sum_inv) = initialize_baum_welch_multiple_sequences(obs_densities)
    return baum_welch_multiple_sequences!(
        obs_densities, α, β, γ, ξ, α_sum_inv, hmm_init, obs_sequences; kwargs...
    )
end

"""
    baum_welch(hmm_init, obs_sequence)

Same as [`baum_welch_multiple_sequences`](@ref) but with a single sequence.
"""
function baum_welch(hmm_init::HMM, obs_sequence::AbstractVector; kwargs...)
    return baum_welch_multiple_sequences(hmm_init, [obs_sequence]; kwargs...)
end
