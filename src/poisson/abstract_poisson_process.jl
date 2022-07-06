abstract type AbstractPoissonProcess end

@inline DensityInterface.DensityKind(::AbstractPoissonProcess) = HasDensity()

function log_intensity end
function ground_intensity end
function mark_distribution end

intensity(pp::AbstractPoissonProcess, args...) = exp(log_intensity(pp, args...))
