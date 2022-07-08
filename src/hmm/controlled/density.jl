function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractMatrix,
    args...,
)
    α, logL = light_forward(hmm, obs_sequence, control_sequence, args...)
    return logL
end
