function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    params;
)
    α, logL = light_forward(hmm, obs_sequence, control_sequence, params)
    return logL
end
