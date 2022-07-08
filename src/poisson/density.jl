function DensityInterface.logdensityof(pp::AbstractPoissonProcess, h::History)
    logL = -ground_intensity(pp) * duration(h)
    for m in event_marks(h)
        logL += log_intensity(pp, m)
    end
    return logL
end
