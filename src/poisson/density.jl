function DensityInterface.logdensityof(pp::AbstractPoissonProcess, h::History)
    l = -ground_intensity(pp) * duration(h)
    for m in event_marks(h)
        l += log_intensity(pp, m)
    end
    return l
end
