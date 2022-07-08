function infer_current_state(hmm::AbstractHMM, obs_sequence::AbstractVector)
    α, logL = light_forward(hmm, obs_sequence)
    return α
end
