function Base.rand(
    rng::AbstractRNG, pp::AbstractPoissonProcess, tmin::Real=0.0, tmax::Real=1.0
)
    mark_dist = mark_distribution(pp)
    N = rand(rng, Poisson(float(ground_intensity(pp) * (tmax - tmin))))
    times = rand(rng, Uniform(tmin, tmax), N)
    marks = [rand(rng, mark_dist) for n in 1:N]
    return History(; times=times, marks=marks, tmin=tmin, tmax=tmax)
end

function Base.rand(pp::AbstractPoissonProcess, tmin::Real=0.0, tmax::Real=1.0)
    return rand(GLOBAL_RNG, pp, tmin, tmax)
end
