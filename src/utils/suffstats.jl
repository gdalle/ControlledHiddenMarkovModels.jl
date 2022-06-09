## Types

suffstats_type(::Type{D}) where {D<:Normal} = Distributions.NormalStats

## Addition

function add_suffstats(ss1::Distributions.NormalStats, ss2::Distributions.NormalStats)
    s = ss1.s + ss2.s
    m = (ss1.tw * ss1.m + ss2.tw * ss2.m) / (ss1.tw + ss2.tw)
    s2 = ss1.s2 + ss2.s2
    tw = ss1.tw + ss2.tw
    return Distributions.NormalStats(s, m, s2, tw)
end
