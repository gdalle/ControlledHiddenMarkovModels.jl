struct DelimitedPoissonProcess{P,T} <: AbstractPoissonProcess
    pp::P
    tmin::T
    tmax::T
end

DelimitedPoissonProcess(pp; tmin, tmax) = DelimitedPoissonProcess(pp, tmin, tmax)

log_intensity(d::DelimitedPoissonProcess, m) = log_intensity(d.pp, m)
ground_intensity(d::DelimitedPoissonProcess) = ground_intensity(d.pp)
mark_distribution(d::DelimitedPoissonProcess) = mark_distribution(d.pp)

function Base.rand(rng::AbstractRNG, d::DelimitedPoissonProcess)
    return rand(rng, d.pp, d.tmin, d.tmax)
end
