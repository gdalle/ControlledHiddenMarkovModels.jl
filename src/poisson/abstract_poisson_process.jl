abstract type AbstractPoissonProcess end

@inline DensityInterface.DensityKind(::AbstractPoissonProcess) = HasDensity()

log_intensity(::AbstractPoissonProcess, m) = error("Not implemented.")
ground_intensity(::AbstractPoissonProcess) = error("Not implemented.")
mark_distribution(::AbstractPoissonProcess) = error("Not implemented.")

intensity(pp::AbstractPoissonProcess, args...) = exp(log_intensity(pp, args...))
