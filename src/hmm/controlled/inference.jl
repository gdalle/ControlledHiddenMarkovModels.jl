function infer_current_state(
    hmm::AbstractControlledHMM,
    obs_sequence::AbstractVector,
    control_matrix::AbstractMatrix,
    ps,
    st,
)
    α, logL = light_forward(hmm, obs_sequence, control_matrix, ps, st)
    return α
end
