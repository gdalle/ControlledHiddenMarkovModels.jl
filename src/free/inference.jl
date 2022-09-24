"""
    logdensityof(hmm::AbstractHMM, obs_sequence, par; safe)

Compute the log likelihood of `obs_sequence` for `hmm` with parameters `par`.
"""
function DensityInterface.logdensityof(hmm::AbstractHMM, obs_sequence, par=nothing; safe=2)
    α, logL = light_forward(obs_sequence, hmm, par)
    return logL
end

"""
    infer_current_state(hmm::AbstractHMM, obs_sequence, par; safe)

Infer the posterior distribution of the current state given `obs_sequence` for `hmm` with parameters `par`.
"""
function infer_current_state(hmm::AbstractHMM, obs_sequence, par=nothing; safe=false)
    α, logL = light_forward(obs_sequence, hmm, par)
    return α
end
