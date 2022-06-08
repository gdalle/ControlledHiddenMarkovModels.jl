Base.length(pp::MultivariatePoissonProcess) = length(pp.λ)

ground_intensity(pp::MultivariatePoissonProcess) = sum(pp.λ)
intensity(pp::MultivariatePoissonProcess, m::Integer) = pp.λ[m]
log_intensity(pp::MultivariatePoissonProcess, m::Integer) = log(pp.λ[m])
mark_distribution(pp::MultivariatePoissonProcess) = pp.mark_dist
