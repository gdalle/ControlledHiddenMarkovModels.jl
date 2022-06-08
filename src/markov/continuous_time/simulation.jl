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
    D = rates_negdiag(mc)
    P = embedded_transition_matrix(mc)
    transitions = [Categorical(P[s, :]; check_args=check_args) for s in 1:nb_states(mc)]
    waiting_times = [Exponential(1 / D[s]; check_args=check_args) for s in 1:nb_states(mc)]
    h = History(; times=R2[], marks=Int[], tmin=tmin, tmax=tmax)
    s = rand(rng, Categorical(mc.p0; check_args=check_args))
    push!(h, tmin, s)
    t = tmin
    while t < tmax
        Δt = rand(rng, waiting_times[s])
        if t + Δt < tmax
            t += Δt
            s = rand(rng, transitions[s])
            push!(h, t, s)
        else
            break
        end
    end
    return h
end

"""
    rand([rng,] prior::ContinuousMarkovChainPrior)

Sample a [`ContinuousMarkovChain`](@ref) from `prior`.
"""
function Base.rand(
    rng::AbstractRNG, prior::ContinuousMarkovChainPrior{R1,R2,R3}; check_args=false
) where {R1<:Real,R2<:Real,R3<:Real}
    return error("not implemented")
end

function Base.rand(mc::ContinuousMarkovChain, tmin::Real, tmax::Real; kwargs...)
    return rand(GLOBAL_RNG, mc, tmin, tmax; kwargs...)
end

Base.rand(prior::ContinuousMarkovChainPrior; kwargs...) = rand(GLOBAL_RNG, prior; kwargs...)
