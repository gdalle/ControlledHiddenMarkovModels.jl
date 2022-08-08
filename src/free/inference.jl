function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_sequence::AbstractVector, par=nothing
)
    α, logL = light_forward(obs_sequence, hmm, par)
    return logL
end

function infer_current_state(hmm::AbstractHMM, obs_sequence::AbstractVector, par=nothing)
    α, logL = light_forward(obs_sequence, hmm, par)
    return α
end
