function initialize_baum_welch_multiple_sequences(
    obs_densities::AbstractVector{<:AbstractMatrix{R}}
) where {R<:Real}
    K = length(obs_densities)
    S = size(obs_densities[1], 1)
    T = [size(obs_densities[k], 2) for k in 1:K]
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    c = [Vector{R}(undef, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    return (α=α, β=β, c=c, γ=γ, ξ=ξ)
end

function baum_welch_multiple_sequences!(
    obs_densities::AbstractVector{<:AbstractMatrix{R}},
    α::AbstractVector{<:AbstractMatrix{R}},
    β::AbstractVector{<:AbstractMatrix{R}},
    c::AbstractVector{<:AbstractVector{R}},
    γ::AbstractVector{<:AbstractMatrix{R}},
    ξ::AbstractVector{<:AbstractArray{R,3}},
    hmm_init::HMM{Tr,Em},
    obs_sequences::AbstractVector;
    max_iterations::Integer=100,
    tol::Real=1e-3,
    show_progress::Bool=true,
    plot::Bool=false,
) where {Tr,Em,R}
    hmm = hmm_init
    S = nb_states(hmm)
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    concat_obs_sequences = reduce(vcat, obs_sequences)

    logL_evolution = float(R)[]
    prog = Progress(max_iterations; desc="Baum-Welch algorithm", enabled=show_progress)
    for iteration in 1:max_iterations
        logL = zero(float(R))
        for k in 1:K
            update_obs_density!(obs_densities[k], hmm, obs_sequences[k])
            logL += forward_backward!(α[k], β[k], c[k], γ[k], ξ[k], hmm, obs_densities[k])
        end
        push!(logL_evolution, logL)

        new_transitions = convert(Tr, fit_mle(Tr, γ, ξ))
        new_emissions = Vector{Em}(undef, S)
        for s in 1:S
            concat_γs = mapreduce(x -> view(x, s, :), vcat, γ)
            concat_γs_float64 = convert(Vector{Float64}, concat_γs)
            new_emissions[s] = convert(
                Em, fit_mle(Em, concat_obs_sequences, concat_γs_float64)
            )
        end

        hmm = HMM(new_transitions, new_emissions)

        if iteration > 1 && (logL_evolution[end] - logL_evolution[end - 1]) / sum(T) < tol
            break
        else
            next!(prog)
        end
    end

    plot && plot_baum_welch(logL_evolution)

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
    (; α, β, c, γ, ξ) = initialize_baum_welch_multiple_sequences(obs_densities)
    return baum_welch_multiple_sequences!(
        obs_densities, α, β, c, γ, ξ, hmm_init, obs_sequences; kwargs...
    )
end

"""
    baum_welch(hmm_init, obs_sequences)

Same as [`baum_welch_multiple_sequences`](@ref) but with a single sequence.
"""
function baum_welch(
    hmm_init::HMM, obs_sequences::AbstractVector; kwargs...
)
    return baum_welch_multiple_sequences(hmm_init, [obs_sequences]; kwargs...)
end
