function add_suffstats(stats1::Distributions.NormalStats, stats2::Distributions.NormalStats)
    s = stats1.s + stats2.s
    s2 = stats1.s2 + stats2.s2
    tw = stats1.tw + stats2.tw
    m = s / tw
    stats = Distributions.NormalStats(s, m, s2, tw)
    return stats
end

function fit_mle_from_multiple_sequences(D::Type{<:Distributions.Normal}, xs, ws)
    stats = reduce(add_suffstats, suffstats(D, x, w) for (x, w) in zip(xs, ws))
    return fit_mle(D, stats)
end
