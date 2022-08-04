function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    params;
    log=false,
)
    if log
        α, logL = light_logforward(hmm, obs_sequence, control_sequence, params)
    else
        α, logL = light_forward(hmm, obs_sequence, control_sequence, params)
    end
    return logL
end
