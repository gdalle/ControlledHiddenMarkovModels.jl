"""
    log_initial_distribution(hmm::AbstractControlledHMM, par)

Return the vector of initial state probabilities _in log scale_ for `hmm` with parameters `par`.
"""
function log_initial_distribution(hmm::H, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    log_transition_matrix!(logP, hmm::AbstractControlledHMM, control, par)

Update `logP` with the state transition matrix _in log scale_ for `hmm` with control `control` and parameters `par`.
"""
function log_transition_matrix!(logP, hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end

"""
    log_transition_matrix(hmm::AbstractControlledHMM, control, par)

Compute the state transition matrix _in log scale_ for `hmm` with control `control` and parameters `par`.
"""
function log_transition_matrix(hmm::H, control, par) where {H<:AbstractControlledHMM}
    return error("Not implemented for type $H")
end
