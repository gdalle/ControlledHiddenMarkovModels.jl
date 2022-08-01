function logsumexp(a)
    m = maximum(a)
    lse = m + log(sum(exp, x - m for x in a))
    return lse
end
