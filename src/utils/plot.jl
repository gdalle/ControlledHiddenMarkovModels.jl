function plot_baum_welch(logL_evolution::AbstractVector{<:Real})
    plt = lineplot(
        logL_evolution;
        title="Baum-Welch convergence",
        xlabel="Iteration",
        ylabel="Log-likelihood",
    )
    println(plt)
end
