"""
    logdensityof(hmm::AbstractHMM, obs_sequence, par; safe)

Compute the log likelihood of `obs_sequence` for `hmm` with parameters `par`.

If `safe = true`, everything is done in log scale.
"""
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

"""
    infer_current_state(hmm::AbstractHMM, obs_sequence, par; safe)

Infer the posterior distribution of the current state given `obs_sequence` for `hmm` with parameters `par`.

If `safe = true`, everything is done in log scale.
"""
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
