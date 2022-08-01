function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix,
    args...,
)
    α, logL = light_forward(hmm, obs_sequence, control_matrix, args...)
    return logL
end
