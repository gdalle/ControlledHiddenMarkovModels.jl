function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    par=nothing;
    safe=false,
)
    if safe
        α, logL = light_forward_log(obs_sequence, control_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    end
    return logL
end

function infer_current_state(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    par=nothing;
    safe=false,
)
    if safe
        α, logL = light_forward_log(obs_sequence, control_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    end
    return α
end
