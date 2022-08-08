struct HiddenMarkovModel{R1,R2,D} <: AbstractHMM
    p0::Vector{R1}
    P::Matrix{R2}
    emissions::Vector{D}
end

const HMM = HiddenMarkovModel

nb_states(hmm::HMM, par=nothing) = length(hmm.p0)
initial_distribution(hmm::HMM, par=nothing) = hmm.p0
transition_matrix(hmm::HMM, par=nothing) = hmm.P
emission_distribution(hmm::HMM, s::Integer, par=nothing) = hmm.emissions[s]

emission_type(::Type{HMM{R1,R2,D}}) where {R1,R2,D} = D

function fit_from_multiple_sequences(::Type{D}, xs, ws) where {D}
    return error(
        "The method `fit_from_multiple_sequences(::Type{D}, xs, ws)` is not implemented for emission type $D. It is required for the Baum-Welch algorithm",
    )
end
