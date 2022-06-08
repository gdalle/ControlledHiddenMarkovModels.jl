function is_rates_mat(Q::AbstractMatrix{R}; atol=1e-5) where {R<:Real}
    n, m = size(Q)
    n == m || return false
    for i in 1:n
        for j in 1:n
            if i != j && Q[i, j] < zero(R)
                return false
            elseif i == j && Q[i, i] > zero(R)
                return false
            end
        end
        if !isapprox(sum(view(Q, i, :)), zero(R); atol=atol)
            return false
        end
    end
    return true
end
