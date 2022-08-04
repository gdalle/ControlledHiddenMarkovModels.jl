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
    logp0 = log_initial_distribution(mc)
    p0 = exp.(logp0)
    c₁ = control_sequence[1]
    logP = log_transition_matrix(mc, c₁, parameters)
    P = exp.(logP)
    for k in 1:nb_samples
        s = rand(rng, Categorical(p0; check_args=check_args))
        if s == target
            hitting_times[k] = 0
        else
            @views for t in 1:(T - 1)
                cₜ = control_sequence[t]
                log_transition_matrix!(logP, mc, cₜ, parameters)
                P .= exp.(logP)
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
