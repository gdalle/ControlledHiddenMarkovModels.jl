function sample_hitting_times(
    rng::AbstractRNG,
    mc::AbstractControlledMarkovChain,
    target::Integer,
    control_sequence::AbstractVector,
    parameters;
    nb_samples=10,
    check_args=false,
)
    T = length(control_sequence)
    hitting_times = fill(T + 1, nb_samples)
    p0 = initial_distribution(mc, parameters)
    c₁ = control_sequence[1]
    P = transition_matrix(mc, c₁, parameters)
    for k in 1:nb_samples
        s = rand(rng, Categorical(p0; check_args=check_args))
        if s == target
            hitting_times[k] = 0
        else
            @views for t in 1:(T - 1)
                cₜ = control_sequence[t]
                transition_matrix!(P, mc, cₜ, parameters)
                s = rand(rng, Categorical(P[s, :]; check_args=check_args))
                if s == target
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
    control_sequence::AbstractVector,
    parameters;
    kwargs...,
)
    return sample_hitting_times(
        GLOBAL_RNG, mc, target, control_sequence, parameters; kwargs...
    )
end
