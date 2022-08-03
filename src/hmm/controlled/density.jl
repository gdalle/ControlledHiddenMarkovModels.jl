function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    params,
)
    logα, logL = light_logforward(hmm, obs_sequence, control_sequence, params)
    return logL
end
