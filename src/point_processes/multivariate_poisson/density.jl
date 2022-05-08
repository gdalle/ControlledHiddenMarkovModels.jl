function DensityInterface.logdensityof(pp::MultivariatePoissonProcess, h::History)
    l = -ground_intensity(pp) * duration(h)
    for m in event_marks(h)
        l += log_intensity(pp, m)
    end
    return l
end

function DensityInterface.logdensityof(
    prior::MultivariatePoissonProcessPrior, pp::MultivariatePoissonProcess
)
    l = sum(
        logdensityof(Gamma(prior.λα[m], inv(prior.λβ[m]); check_args=false), pp.λ[m]) for
        m in 1:length(pp)
    )
    return l
end
