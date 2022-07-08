abstract type AbstractPoissonProcess end

@inline DensityInterface.DensityKind(::AbstractPoissonProcess) = HasDensity()

log_intensity(::AbstractPoissonProcess, m) = error("not implemented")
ground_intensity(::AbstractPoissonProcess) = error("not implemented")
mark_distribution(::AbstractPoissonProcess) = error("not implemented")

intensity(pp::AbstractPoissonProcess, args...) = exp(log_intensity(pp, args...))
