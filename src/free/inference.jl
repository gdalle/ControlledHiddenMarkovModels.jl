function DensityInterface.logdensityof(
    hmm::AbstractHMM, obs_sequence::AbstractVector, par=nothing; safe=false
)
    if safe
        α, logL = light_forward_log(obs_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, hmm, par)
    end
    return logL
end

function infer_current_state(
    hmm::AbstractHMM, obs_sequence::AbstractVector, par=nothing; safe=false
)
    if safe
        α, logL = light_forward_log(obs_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, hmm, par)
    end
    return α
end
