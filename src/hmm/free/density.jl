function DensityInterface.logdensityof(hmm::AbstractHMM, obs_sequence::AbstractVector)
    α, logL = light_forward(hmm, obs_sequence)
    return logL
end
