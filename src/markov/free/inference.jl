function sample_hitting_times(
    rng::AbstractRNG,
    mc::MarkovChain,
    target::Integer,
    T::Integer,
    p0=initial_distribution(mc);
    nb_samples=10,
    check_args=false,
)
    P = transition_matrix(mc)
    hitting_times = fill(T + 1, nb_samples)
    for k in 1:nb_samples
        i = rand(rng, Categorical(p0; check_args=check_args))
        if i == target
            hitting_times[k] = 0
        else
            for t in 1:(T - 1)
                P_row = @view P[i, :]
                i = rand(rng, Categorical(P_row; check_args=check_args))
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
    mc::MarkovChain, target::Integer, T::Integer, p0=initial_distribution(mc); kwargs...
)
    return sample_hitting_times(GLOBAL_RNG, mc, target, T, p0; kwargs...)
end