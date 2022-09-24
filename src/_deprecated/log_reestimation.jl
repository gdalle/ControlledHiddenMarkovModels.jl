function initial_distribution(log_fb_storage::LogForwardBackwardStorage)
    (; logγ) = log_fb_storage
    @views p0 = reduce(+, exp.(logγ[k][:, 1]) for k in eachindex(logγ))  # TODO: fix alloc
    p0 ./= sum(p0)
    return p0
end

function transition_matrix(log_fb_storage::LogForwardBackwardStorage)
    (; logξ) = log_fb_storage
    P = reduce(+, dropdims(sum(exp, logξ[k]; dims=3); dims=3) for k in eachindex(logξ))
    P ./= sum(P; dims=2)
    return P
end

function emission_distribution(
    ::Type{H},
    log_fb_storage::LogForwardBackwardStorage,
    obs_sequences::AbstractVector{<:AbstractVector},
    s::Integer,
) where {H<:AbstractHMM}
    (; logγ) = log_fb_storage
    D = emission_type(H)
    xs = (obs_sequences[k] for k in eachindex(obs_sequences))
    ws = (exp.(logγ[k][s, :]) for k in eachindex(logγ))  # TODO: fix alloc
    return fit_mle_from_multiple_sequences(D, xs, ws)
end
