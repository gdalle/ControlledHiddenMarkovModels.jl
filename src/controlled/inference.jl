"""
    logdensityof(hmm::AbstractControlledHMM, obs_sequence, control_sequence, par; safe)

Compute the log likelihood of `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM, obs_sequence, control_sequence, par=nothing; safe=true
)
    if safe
        α, logL = light_forward_log(obs_sequence, control_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    end
    return logL
end

"""
    infer_current_state(hmm::AbstractControlledHMM, obs_sequence, control_sequence, par; safe)

Infer the posterior distribution of the current state given `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function infer_current_state(
    hmm::AbstractControlledHMM, obs_sequence, control_sequence, par=nothing; safe=true
)
    if safe
        α, logL = light_forward_log(obs_sequence, control_sequence, hmm, par)
    else
        α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    end
    return α
end
