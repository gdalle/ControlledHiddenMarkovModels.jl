function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix,
    ps,
    st,
)
    logα, logL = light_logforward(hmm, obs_sequence, control_matrix, ps, st)
    return logL
end
