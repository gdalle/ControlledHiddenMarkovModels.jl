function baum_welch_multiple_sequences!(
    obs_densities::AbstractVector{<:AbstractMatrix{R}},
    fb_storage::ForwardBackwardStorage{R},
    hmm_init::AbstractHMM,
    obs_sequences::AbstractVector,
    control_sequences::AbstractVector=[Fill(nothing, length(seq)) for seq in obs_sequences],
    ps=nothing,
    st=nothing;
    max_iterations::Integer=100,
    opt=Optimisers.Adam(),
    opt_steps::Integer=10,
    tol::Real=1e-3,
    show_progress::Bool=true,
) where {R}
    hmm = hmm_init
    K = length(obs_sequences)
    T = [length(obs_sequences[k]) for k in 1:K]
    (; α, β, γ, ξ, α_sum_inv) = fb_storage

    # Initialize loglikelihood storage
    logL_evolution = float(R)[]
    logL_by_seq = Vector{float(R)}(undef, K)

    prog = Progress(max_iterations; desc="Baum-Welch algorithm", enabled=show_progress)
    for iteration in 1:max_iterations
        let hmm = hmm
            for k in 1:K
                # Local forward-backward
                update_obs_density!(
                    obs_densities[k], hmm, obs_sequences[k], control_sequences[k], ps, st
                )
                logL_by_seq[k] = forward_backward!(
                    α[k],
                    β[k],
                    γ[k],
                    ξ[k],
                    α_sum_inv[k],
                    hmm,
                    obs_densities[k],
                    control_sequences[k],
                    ps,
                    st,
                )
            end
        end
        push!(logL_evolution, sum(logL_by_seq))

        st_opt = Optimisers.setup(opt, ps)
        for _ in 1:opt_steps
            gs_tup = gradient(ps) do ps_local
                sum(
                    forwarddiff(
                        Q_function,
                        hmm,
                        γ[k],
                        ξ[k],
                        obs_sequences[k],
                        control_sequences[k],
                        ps_local,
                        st,
                    ) for k in 1:K
                )
            end
            st_opt, ps = Optimisers.update(st_opt, ps, gs_tup[1])
        end

        if iteration > 1 && (logL_evolution[end] - logL_evolution[end - 1]) / sum(T) < tol
            break
        else
            next!(prog)
        end
    end

    return ps, logL_evolution
end

"""
    baum_welch_multiple_sequences(hmm_init, obs_sequences)

Run the Baum-Welch algorithm to estimate an [`AbstractHiddenMarkovModel`](@ref) of the same type as `hmm_init`, based on several observation sequences.
"""
function baum_welch_multiple_sequences(
    hmm_init::AbstractHMM,
    obs_sequences::AbstractVector,
    control_sequences::AbstractVector=[Fill(nothing, length(seq)) for seq in obs_sequences],
    ps=nothing,
    st=nothing;
    kwargs...,
)
    K = length(obs_sequences)
    obs_densities = [
        compute_obs_density(hmm_init, obs_sequences[k], control_sequences[k], ps, st) for
        k in 1:K
    ]
    fb_storage = initialize_forward_backward_multiple_sequences(obs_densities)
    result = baum_welch_multiple_sequences!(
        obs_densities,
        fb_storage,
        hmm_init,
        obs_sequences,
        control_sequences,
        ps,
        st;
        kwargs...,
    )
    return result
end

"""
    baum_welch(hmm_init, obs_sequence)

Same as [`baum_welch_multiple_sequences`](@ref) but with a single sequence.
"""
function baum_welch(
    hmm_init::AbstractHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector=Fill(nothing, length(obs_sequence)),
    args...;
    kwargs...,
)
    return baum_welch_multiple_sequences(
        hmm_init, [obs_sequence], [control_sequence], args...; kwargs...
    )
end
