function sample_hitting_times(
    rng::AbstractRNG,
    mc::AbstractControlledMarkovChain,
    target::Integer,
    control_matrix::AbstractMatrix,
    p0=initial_distribution(mc);
    nb_samples=10,
    check_args=false,
)
    P_all = transition_matrix(mc, control_matrix, args...)
    hitting_times = fill(T + 1, nb_samples)
    T = size(control_matrix, 2)
    for k in 1:nb_samples
        i = rand(rng, Categorical(p0; check_args=check_args))
        if i == target
            hitting_times[k] = 0
        else
            for t in 1:(T - 1)
                Pₜ_row = view(P_all, i, :, t)
                i = rand(rng, Categorical(Pₜ_row; check_args=check_args))
                if i == target
                    hitting_times[k] = t - 1
                    break
                end
            end
        end
    end
    return hitting_times
end

function sample_hitting_times(
    mc::AbstractControlledMarkovChain,
    target::Integer,
    control_matrix::AbstractMatrix,
    p0=initial_distribution(mc);
    kwargs...,
)
    return sample_hitting_times(GLOBAL_RNG, mc, target, control_matrix, p0; kwargs...)
end
