## Obs densities

struct ObsDensityStorage{R}
    obs_densities::Vector{Matrix{R}}
end

struct LogObsDensityStorage{R}
    obs_logdensities::Vector{Matrix{R}}
end

const AnyObsDensityStorage{R} = Union{ObsDensityStorage{R},LogObsDensityStorage{R}}

## Access

Base.length(od_storage::ObsDensityStorage) = length(od_storage.obs_densities)
Base.length(log_od_storage::LogObsDensityStorage) = length(log_od_storage.obs_logdensities)

function nb_states(od_storage::ObsDensityStorage)
    return size(first(od_storage.obs_densities), 1)
end

function nb_states(log_od_storage::LogObsDensityStorage)
    return size(first(log_od_storage.obs_logdensities), 1)
end

function sequence_durations(od_storage::ObsDensityStorage)
    return [
        size(od_storage.obs_densities[k], 2) for k in eachindex(od_storage.obs_densities)
    ]
end

function sequence_durations(log_od_storage::LogObsDensityStorage)
    return [
        size(log_od_storage.obs_logdensities[k], 2) for
        k in eachindex(log_od_storage.obs_logdensities)
    ]
end

function get_element(od_storage::ObsDensityStorage)
    return first(first(od_storage.obs_densities))
end

function get_element(log_od_storage::LogObsDensityStorage)
    return first(first(log_od_storage.obs_logdensities))
end

function get_normaltype(od_storage::ObsDensityStorage)
    return typeof(get_element(od_storage))
end

function get_normaltype(log_od_storage::LogObsDensityStorage)
    return typeof(exp(get_element(log_od_storage)))
end

function get_logtype(od_storage::ObsDensityStorage)
    return typeof(log(get_element(od_storage)))
end

function get_logtype(log_od_storage::LogObsDensityStorage)
    return typeof(get_element(log_od_storage))
end

## Updates

function update_obs_densities_generic!(
    od_storage::ObsDensityStorage,
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm::AbstractHMM,
    par,
)
    (; obs_densities) = od_storage
    for k in eachindex(obs_densities)
        update_obs_density!(obs_densities[k], obs_sequences[k], hmm, par)
    end
    return nothing
end

function update_obs_densities_generic!(
    log_od_storage::LogObsDensityStorage,
    obs_sequences::AbstractVector{<:AbstractVector},
    hmm::AbstractHMM,
    par,
)
    (; obs_logdensities) = log_od_storage
    for k in eachindex(obs_logdensities)
        update_obs_logdensity!(obs_logdensities[k], obs_sequences[k], hmm, par)
    end
    return nothing
end

## Initialization

function initialize_obs_densities(
    obs_sequences::AbstractVector{<:AbstractVector}, hmm::AbstractHMM, par
)
    obs_densities = [
        initialize_obs_density(obs_sequences[k], hmm, par) for k in eachindex(obs_sequences)
    ]
    return ObsDensityStorage(obs_densities)
end

function initialize_obs_logdensities(
    obs_sequences::AbstractVector{<:AbstractVector}, hmm::AbstractHMM, par
)
    obs_logdensities = [
        initialize_obs_logdensity(obs_sequences[k], hmm, par) for
        k in eachindex(obs_sequences)
    ]
    return LogObsDensityStorage(obs_logdensities)
end

## Forward-backward sufficient stats

"""
    ForwardBackwardStorage{R}

Storage for the forward-backward algorithm applied to multiple sequences.

# Fields

- `α::Vector{Matrix{R}}`
- `c::Vector{Vector{R}}`
- `β::Vector{Matrix{R}}`
- `bβ::Vector{Matrix{R}}`
- `γ::Vector{Matrix{R}}`
- `ξ::Vector{Array{R,3}}`
"""
struct ForwardBackwardStorage{R}
    α::Vector{Matrix{R}}
    c::Vector{Vector{R}}
    β::Vector{Matrix{R}}
    bβ::Vector{Matrix{R}}
    γ::Vector{Matrix{R}}
    ξ::Vector{Array{R,3}}
end

"""
    LogForwardBackwardStorage{R}

Storage for the forward-backward algorithm _in log scale_ applied to multiple sequences.

# Fields

- `logα::Vector{Matrix{R}}`
- `logβ::Vector{Matrix{R}}`
- `logγ::Vector{Matrix{R}}`
- `logξ::Vector{Array{R,3}}`
"""
struct LogForwardBackwardStorage{R}
    logα::Vector{Matrix{R}}
    logβ::Vector{Matrix{R}}
    logγ::Vector{Matrix{R}}
    logξ::Vector{Array{R,3}}
end

const AnyForwardBackwardStorage{R} = Union{
    ForwardBackwardStorage{R},LogForwardBackwardStorage{R}
}

function initialize_forward_backward(od_storage::ObsDensityStorage)
    R = get_normaltype(od_storage)
    K = length(od_storage)
    S = nb_states(od_storage)
    T = sequence_durations(od_storage)
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    c = [Vector{R}(undef, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    bβ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    fb_storage = ForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return fb_storage
end

function initialize_forward_backward(log_od_storage::LogObsDensityStorage)
    R = get_normaltype(log_od_storage)
    K = length(log_od_storage)
    S = nb_states(log_od_storage)
    T = sequence_durations(log_od_storage)
    α = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    c = [Vector{R}(undef, T[k]) for k in 1:K]
    β = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    bβ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    γ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    ξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    fb_storage = ForwardBackwardStorage{R}(α, c, β, bβ, γ, ξ)
    return fb_storage
end

function initialize_forward_backward_log(log_od_storage::LogObsDensityStorage)
    R = get_logtype(log_od_storage)
    K = length(log_od_storage)
    S = nb_states(log_od_storage)
    T = sequence_durations(log_od_storage)
    logα = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    logβ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    logγ = [Matrix{R}(undef, S, T[k]) for k in 1:K]
    logξ = [Array{R,3}(undef, S, S, T[k] - 1) for k in 1:K]
    log_fb_storage = LogForwardBackwardStorage{R}(logα, logβ, logγ, logξ)
    return log_fb_storage
end

function forward_backward_generic!(
    fb_storage::ForwardBackwardStorage, od_storage::ObsDensityStorage, hmm::AbstractHMM, par
)
    (; α, c, β, bβ, γ, ξ) = fb_storage
    (; obs_densities) = od_storage
    logL = zero(get_logtype(od_storage))
    for k in eachindex(obs_densities)
        logL += forward_backward_nolog!(
            α[k], c[k], β[k], bβ[k], γ[k], ξ[k], obs_densities[k], hmm, par
        )
    end
    return logL
end

function forward_backward_generic!(
    fb_storage::ForwardBackwardStorage,
    log_od_storage::LogObsDensityStorage,
    hmm::AbstractHMM,
    par,
)
    (; α, c, β, bβ, γ, ξ) = fb_storage
    (; obs_logdensities) = log_od_storage
    logL = zero(get_logtype(log_od_storage))
    for k in eachindex(obs_logdensities)
        logL += forward_backward_log!(
            α[k], c[k], β[k], bβ[k], γ[k], ξ[k], obs_logdensities[k], hmm, par
        )
    end
    return logL
end

function forward_backward_generic!(
    log_fb_storage::LogForwardBackwardStorage,
    log_od_storage::LogObsDensityStorage,
    hmm::AbstractHMM,
    par,
)
    (; logα, logβ, logγ, logξ) = log_fb_storage
    (; obs_logdensities) = log_od_storage
    logL = zero(get_logtype(log_od_storage))
    for k in eachindex(obs_logdensities)
        logL += forward_backward_doublelog!(
            logα[k], logβ[k], logγ[k], logξ[k], obs_logdensities[k], hmm, par
        )
    end
    return logL
end
