function Base.rand(rng::AbstractRNG, pp::MultivariatePoissonProcess, tmin::Real, tmax::Real)
    mark_dist = mark_distribution(pp)
    N = rand(rng, Poisson(ground_intensity(pp) * (tmax - tmin)))
    times = rand(rng, Uniform(tmin, tmax), N)
    marks = [rand(rng, mark_dist) for n in 1:N]
    return History(; times=times, marks=marks, tmin=tmin, tmax=tmax)
end

function Base.rand(pp::MultivariatePoissonProcess, tmin::Real, tmax::Real)
    return rand(GLOBAL_RNG, pp, tmin, tmax)
end
