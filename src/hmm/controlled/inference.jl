function infer_current_state(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_sequence::AbstractVector,
    parameters;
)
    α, logL = light_forward(hmm, obs_sequence, control_sequence, parameters)
    return α
end
