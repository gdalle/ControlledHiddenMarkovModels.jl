"""
    log_initial_distribution(hmm::AbstractHMM, par)

Compute the vector of initial state probabilities _in log scale_ for `hmm` with parameters `par`.
"""
function log_initial_distribution(hmm::H, par) where {H<:AbstractHMM}
    return log.(initial_distribution(hmm, par))
end

"""
    log_transition_matrix(hmm::AbstractHMM, par)

Compute the state transition matrix _in log scale_ for `hmm` with parameters `par`.
"""
function log_transition_matrix(hmm::H, par) where {H<:AbstractHMM}
    return log.(transition_matrix(hmm, par))
end
