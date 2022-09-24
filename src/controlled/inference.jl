"""
    logdensityof(hmm::AbstractControlledHMM, obs_sequence, control_sequence, par)

Compute the log likelihood of `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM, obs_sequence, control_sequence, par=nothing;
)
    α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    return logL
end

"""
    infer_current_state(hmm::AbstractControlledHMM, obs_sequence, control_sequence, par)

Infer the posterior distribution of the current state given `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function infer_current_state(
    hmm::AbstractControlledHMM, obs_sequence, control_sequence, par=nothing;
)
    α, logL = light_forward(obs_sequence, control_sequence, hmm, par)
    return α
end
