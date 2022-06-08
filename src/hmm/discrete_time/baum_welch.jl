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
                for s in 1:S
                    x = obs_sequences[k]
                    w = view(γ[k], s, :)  # Distributions.jl doesn't allow this for type of emissions, see #1560
                    emissions_suffstats_by_seq[k, s] = suffstats(Em, x, w)
                end
            end
        end
        push!(logL_evolution, sum(logL_by_seq))

        # Aggregated transitions
        transitions_suffstats_agg = reduce(
            add_suffstats, transitions_suffstats_by_seq
        )
        new_transitions = fit_mle(Tr, transitions_suffstats_agg)
        # Aggregated emissions
        new_emissions = Vector{Em}(undef, S)
        for s in 1:S
            emissions_suffstats_agg_s = reduce(
                add_suffstats, view(emissions_suffstats_by_seq, :, s)
            )
            new_emissions[s] = fit_mle(Em, emissions_suffstats_agg_s)
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
