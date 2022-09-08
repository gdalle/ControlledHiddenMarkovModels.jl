"""
    logdensityof(hmm::AbstractControlledHMM, obs_sequence, control_sequence, par; safe)

Compute the log likelihood of `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function DensityInterface.logdensityof(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    par=nothing;
    safe=2,
)
    α, logL = light_forward(obs_sequence, control_sequence, hmm, par; safe=safe)
    return logL
end

"""
    infer_current_state(
        hmm::AbstractControlledHMM, obs_sequence, control_sequence, par; safe
    )

Infer the posterior distribution of the current state given `obs_sequence` for `hmm` with controls `control_sequence` and parameters `par`.
"""
function infer_current_state(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    par=nothing;
    safe=2,
)
    α, logL = light_forward(obs_sequence, control_sequence, hmm, par; safe=safe)
    return α
end
