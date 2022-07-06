"""
    rand([rng,] mc::ContinuousMarkovChain, tmin, tmax)

Simulate `mc` on interval `[tmin, tmax)`.
"""
function Base.rand(
    rng::AbstractRNG,
    mc::ContinuousMarkovChain{R1,R2},
    tmin::Real,
    tmax::Real;
    check_args=false,
) where {R1<:Real,R2<:Real}
    S = nb_states(mc)
    p0 = initial_distribution(mc)
    P = embedded_transition_matrix(mc)
    D = intensity_negdiag(mc)
    transitions = [Categorical(view(P, i, :); check_args=check_args) for i in 1:S]
    waiting_times = [Exponential(1 / D[i]; check_args=check_args) for i in 1:S]
    h = History(; times=R2[], marks=Int[], tmin=tmin, tmax=tmax)
    i = rand(rng, Categorical(p0; check_args=check_args))
    push!(h, tmin, i)
    t = tmin
    while t < tmax
        Δt = rand(rng, waiting_times[i])
        if t + Δt < tmax
            t += Δt
            i = rand(rng, transitions[i])
            push!(h, t, i)
        else
            break
        end
    end
    return h
end
