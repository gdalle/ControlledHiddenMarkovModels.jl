function initial_distribution(fb_storage::ForwardBackwardStorage)
    (; γ) = fb_storage
    @views p0 = reduce(+, γ[k][:, 1] for k in eachindex(γ))
    p0 ./= sum(p0)
    return p0
end

function initial_distribution(log_fb_storage::LogForwardBackwardStorage)
    (; logγ) = log_fb_storage
    @views p0 = reduce(+, exp.(logγ[k][:, 1]) for k in eachindex(logγ))  # TODO: fix alloc
    p0 ./= sum(p0)
    return p0
end

## Transition

function transition_matrix(fb_storage::ForwardBackwardStorage)
    (; ξ) = fb_storage
    P = reduce(+, dropdims(sum(ξ[k]; dims=3); dims=3) for k in eachindex(ξ))
    P ./= sum(P; dims=2)
    return P
end

function transition_matrix(log_fb_storage::LogForwardBackwardStorage)
    (; logξ) = log_fb_storage
    P = reduce(+, dropdims(sum(exp, logξ[k]; dims=3); dims=3) for k in eachindex(logξ))
    P ./= sum(P; dims=2)
    return P
end

## Emissions

function emission_distribution(
    ::Type{H}, fb_storage::ForwardBackwardStorage, obs_sequences, s
) where {H<:AbstractHMM}
    (; γ) = fb_storage
    D = emission_type(H)
    S = size(γ[1], 1)
    xs = (obs_sequences[k] for k in eachindex(obs_sequences))
    ws = (γ[k][s, :] for k in eachindex(γ))
    return fit_mle_from_multiple_sequences(D, xs, ws)
end

function emission_distribution(
    ::Type{H}, log_fb_storage::LogForwardBackwardStorage, obs_sequences, s
) where {H<:AbstractHMM}
    (; logγ) = log_fb_storage
    D = emission_type(H)
    xs = (obs_sequences[k] for k in eachindex(obs_sequences))
    ws = (exp.(logγ[k][s, :]) for k in eachindex(logγ))  # TODO: fix alloc
    return fit_mle_from_multiple_sequences(D, xs, ws)
end
