var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [HiddenMarkovModels]","category":"page"},{"location":"api/#Full-docs","page":"API reference","title":"Full docs","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [HiddenMarkovModels]","category":"page"},{"location":"api/#HiddenMarkovModels.DiscreteMarkovChain","page":"API reference","title":"HiddenMarkovModels.DiscreteMarkovChain","text":"DiscreteMarkovChain\n\nDiscrete-time Markov chain with finite state space.\n\nFields\n\np0::AbstractVector: initial state distribution.\nP::AbstractMatrix: state transition matrix.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.DiscreteMarkovChainPrior","page":"API reference","title":"HiddenMarkovModels.DiscreteMarkovChainPrior","text":"DiscreteMarkovChainPrior\n\nDefine a Dirichlet prior on the initial distribution and on the transition matrix of a DiscreteMarkovChain.\n\nFields\n\np0_α::AbstractVector: Dirichlet parameter for the initial distribution\nP_α::AbstractMatrix: Dirichlet parameters for the transition matrix\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.DiscreteMarkovChainStats","page":"API reference","title":"HiddenMarkovModels.DiscreteMarkovChainStats","text":"DiscreteMarkovChainStats\n\nStore sufficient statistics for the likelihood of a DiscreteMarkovChain sample.\n\nFields\n\ninitialization_count::AbstractVector: count initializations in each state\ntransition_count::AbstractMatrix: count transitions between each pair of states\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.HMM","page":"API reference","title":"HiddenMarkovModels.HMM","text":"HMM\n\nAlias for HiddenMarkovModel.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.HMMPrior","page":"API reference","title":"HiddenMarkovModels.HMMPrior","text":"HMMPrior\n\nAlias for HiddenMarkovModelPrior.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.HiddenMarkovModel","page":"API reference","title":"HiddenMarkovModels.HiddenMarkovModel","text":"HiddenMarkovModel{Tr,Em}\n\nHidden Markov Model with arbitrary transition model (must be a discrete Markov chain) and emission distributions.\n\nFields\n\ntransitions::Tr: state evolution process.\nemissions::Vector{Em}: one emission distribution per state.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.HiddenMarkovModelPrior","page":"API reference","title":"HiddenMarkovModels.HiddenMarkovModelPrior","text":"HiddenMarkovModelPrior{TrP,EmP}\n\nPrior for a HiddenMarkovModel.\n\nFields\n\ntransitions_prior::TrP: prior on the transition structure.\nemissions_prior::Vector{EmP}: one prior per state emission distribution.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.History","page":"API reference","title":"HiddenMarkovModels.History","text":"History{M,T<:Real}\n\nLinear event histories with marks of type M and locations of real type T.\n\nFields\n\ntimes::Vector{T}: vector of event times\nmarks::Vector{M}: vector of event marks\ntmin::T: start time\ntmax::T: end time\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.MultivariatePoissonProcess","page":"API reference","title":"HiddenMarkovModels.MultivariatePoissonProcess","text":"MultivariatePoissonProcess{R}\n\nMultivariate homogeneous temporal Poisson process.\n\nFields\n\nλ::Vector{R}: event rates.\n\n\n\n\n\n","category":"type"},{"location":"api/#Base.append!-Tuple{History, History}","page":"API reference","title":"Base.append!","text":"append!(h1::History, h2::History)\n\nAdd all the events of h2 at the end of h1.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.push!-Tuple{History, Any, Any}","page":"API reference","title":"Base.push!","text":"push!(h::History, t, m)\n\nAdd event (t, m) at the end of history h.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{Random.AbstractRNG, DiscreteMarkovChain, Integer}","page":"API reference","title":"Base.rand","text":"rand([rng,] mc::DiscreteMarkovChain, T)\n\nSimulate mc during T time steps.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{Random.AbstractRNG, DiscreteMarkovChainPrior}","page":"API reference","title":"Base.rand","text":"rand([rng,] prior::DiscreteMarkovChainPrior)\n\nSample a DiscreteMarkovChain from prior.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{Random.AbstractRNG, HiddenMarkovModel, Integer}","page":"API reference","title":"Base.rand","text":"rand([rng,] hmm::HMM, T)\n\nSample a sequence of states of length T and the associated sequence of observations.\n\n\n\n\n\n","category":"method"},{"location":"api/#DensityInterface.logdensityof-Tuple{DiscreteMarkovChain, AbstractVector}","page":"API reference","title":"DensityInterface.logdensityof","text":"logdensityof(mc::DiscreteMarkovChain, x::AbstractVector)\n\nCompute the log-likelihood of the sequence x of states for the chain mc.\n\n\n\n\n\n","category":"method"},{"location":"api/#DensityInterface.logdensityof-Tuple{DiscreteMarkovChainPrior, DiscreteMarkovChain}","page":"API reference","title":"DensityInterface.logdensityof","text":"logdensityof(prior::DiscreteMarkovChainPrior, mc::DiscreteMarkovChain)\n\nCompute the log-likelihood of the chain mc with respect to a prior.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.baum_welch-Tuple{HiddenMarkovModel, AbstractVector}","page":"API reference","title":"HiddenMarkovModels.baum_welch","text":"baum_welch(hmm_init, obs_sequence)\n\nSame as baum_welch_multiple_sequences but with a single sequence.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.baum_welch_multiple_sequences-Tuple{HiddenMarkovModel, AbstractVector}","page":"API reference","title":"HiddenMarkovModels.baum_welch_multiple_sequences","text":"baum_welch_multiple_sequences(hmm_init, obs_sequences)\n\nRun the Baum-Welch algorithm to estimate a HiddenMarkovModel of the same type as hmm_init, based on several observation sequences.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.duration-Tuple{History}","page":"API reference","title":"HiddenMarkovModels.duration","text":"duration(h::History)\n\nCompute the difference h.tmax - h.tmin.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.flat_prior-Union{Tuple{DiscreteMarkovChain{R1, R2}}, Tuple{R2}, Tuple{R1}} where {R1<:Real, R2<:Real}","page":"API reference","title":"HiddenMarkovModels.flat_prior","text":"zero_prior(mc::DiscreteMarkovChain)\n\nBuild a flat prior, for which MAP is equivalent to MLE.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.forward_backward!-Union{Tuple{R}, Tuple{AbstractMatrix{R}, AbstractMatrix{R}, AbstractMatrix{R}, AbstractArray{R, 3}, AbstractVector{R}, AbstractVector{R}, AbstractVector{R}, HiddenMarkovModel, AbstractMatrix{R}}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.forward_backward!","text":"forward_backward!(α, β, γ, ξ, α_sum, γ_sum, ξ_sum, hmm, obs_density)\n\nApply the forward-backward algorithm in-place to update sufficient statistics.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.has_events","page":"API reference","title":"HiddenMarkovModels.has_events","text":"has_events(h::History, tmin=-Inf, tmax=Inf)\n\nCheck the presence of events in h during the interval [tmin, tmax).\n\n\n\n\n\n","category":"function"},{"location":"api/#HiddenMarkovModels.initial_distribution-Tuple{DiscreteMarkovChain}","page":"API reference","title":"HiddenMarkovModels.initial_distribution","text":"initial_distribution(mc::DiscreteMarkovChain)\n\nReturn the vector of initial state probabilities of mc.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.max_mark-Union{Tuple{History{M}}, Tuple{M}} where M","page":"API reference","title":"HiddenMarkovModels.max_mark","text":"max_mark(h::History)\n\nCompute the highest mark, provided they can be ordered.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.max_time-Tuple{History}","page":"API reference","title":"HiddenMarkovModels.max_time","text":"max_time(h)\n\nReturn the end time of h (not the same as the last event time).\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.min_time-Tuple{History}","page":"API reference","title":"HiddenMarkovModels.min_time","text":"min_time(h)\n\nReturn the starting time of h (not the same as the first event time).\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.nb_events","page":"API reference","title":"HiddenMarkovModels.nb_events","text":"nb_events(h::History, tmin=-Inf, tmax=Inf)\n\nCount events in h during the interval [tmin, tmax).\n\n\n\n\n\n","category":"function"},{"location":"api/#HiddenMarkovModels.nb_states-Tuple{DiscreteMarkovChain}","page":"API reference","title":"HiddenMarkovModels.nb_states","text":"nb_states(mc::DiscreteMarkovChain)\n\nReturn the number of states of mc.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.rand_prob_vec-Union{Tuple{R}, Tuple{Type{R}, Integer}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.rand_prob_vec","text":"rand_prob_vec(n)\n\nReturn a random probability distribution vector of size n.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.rand_trans_mat-Union{Tuple{R}, Tuple{Type{R}, Integer}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.rand_trans_mat","text":"rand_trans_mat(n)\n\nReturn a stochastic matrix of size n with random transition probability distributions.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.split_into_chunks-Union{Tuple{M}, Tuple{History{M}, Any}} where M","page":"API reference","title":"HiddenMarkovModels.split_into_chunks","text":"split_into_chunks(h, chunk_duration)\n\nSplit h into a vector of consecutive histories with individual duration chunk_duration.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.stationary_distribution-Tuple{DiscreteMarkovChain}","page":"API reference","title":"HiddenMarkovModels.stationary_distribution","text":"stationary_distribution(mc::DiscreteMarkovChain)\n\nCompute the equilibrium distribution of mc using its eigendecomposition.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.time_change-Tuple{History, Any}","page":"API reference","title":"HiddenMarkovModels.time_change","text":"time_change(h, Λ)\n\nApply the time rescaling t -> Λ(t) to history h.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.transition_matrix-Tuple{DiscreteMarkovChain}","page":"API reference","title":"HiddenMarkovModels.transition_matrix","text":"transition_matrix(mc::DiscreteMarkovChain)\n\nReturn the matrix of transition probabilities of mc.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.uniform_prob_vec-Union{Tuple{R}, Tuple{Type{R}, Integer}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.uniform_prob_vec","text":"uniform_prob_vec(n)\n\nReturn a uniform probability distribution vector of size n.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.uniform_trans_mat-Union{Tuple{R}, Tuple{Type{R}, Integer}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.uniform_trans_mat","text":"uniform_trans_mat(n)\n\nReturn a stochastic matrix of size n with uniform transition probability distributions.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.update_obs_density!-Union{Tuple{R}, Tuple{AbstractMatrix{R}, HiddenMarkovModel, AbstractVector}} where R<:Real","page":"API reference","title":"HiddenMarkovModels.update_obs_density!","text":"update_obs_density!(obs_density, hmm, obs_sequence)\n\nSet obs_density[s, t] to the likelihood of hmm emitting obs_sequence[t] if it were in state s.\n\n\n\n\n\n","category":"method"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"EditURL = \"https://github.com/gdalle/HiddenMarkovModels.jl/blob/main/test/examples/discrete_markov.jl\"","category":"page"},{"location":"examples/discrete_markov/#Discrete-Markov-chain","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"","category":"section"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"using HiddenMarkovModels\nusing LogarithmicNumbers\nusing Plots\nusing Statistics","category":"page"},{"location":"examples/discrete_markov/#Construction","page":"Discrete Markov chain","title":"Construction","text":"","category":"section"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"A DiscreteMarkovChain object is built by combining a vector of initial probabilities with a transition matrix.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"p0 = [0.3, 0.7]\nP = [0.9 0.1; 0.2 0.8]\nmc = DiscreteMarkovChain(p0, P)","category":"page"},{"location":"examples/discrete_markov/#Simulation","page":"Discrete Markov chain","title":"Simulation","text":"","category":"section"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"To simulate it, we only need to decide how long the sequence should be.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"state_sequence = rand(mc, 1000);\nnothing #hide","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"Let us visualize the sequence of states.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"scatter(\n    state_sequence;\n    title=\"Markov chain evolution\",\n    xlabel=\"Time\",\n    ylabel=\"State\",\n    label=nothing,\n    margin=5Plots.mm\n)","category":"page"},{"location":"examples/discrete_markov/#Learning","page":"Discrete Markov chain","title":"Learning","text":"","category":"section"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"Based on a sequence of states, we can fit a DiscreteMarkovChain with Maximum Likelihood Estimation (MLE). To speed up estimation, we can specify the types of the parameters to estimate, for instance Float32 instead of Float64.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"mc_mle = fit_mle(DiscreteMarkovChain{Float32,Float32}, state_sequence)","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"As we can see, the error on the transition matrix is quite small.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"error_mle = mean(abs, transition_matrix(mc_mle) - transition_matrix(mc))","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"We can also use a Maximum A Posteriori (MAP) approach by specifying a conjugate prior, which contains observed pseudocounts of intializations and transitions. Let's say we have previously observed 4 trajectories of length 10, with balanced initializations and transitions.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"p0_α = Float32.(1 .+ 4 * [0.5, 0.5])\nP_α = Float32.(1 .+ 4 * 10 * [0.5 0.5; 0.5 0.5])\nmc_prior = DiscreteMarkovChainPrior(p0_α, P_α)\n\nmc_map = fit_map(DiscreteMarkovChain{Float32,Float32}, mc_prior, state_sequence)","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"This results in an estimate that puts larger weights on transitions between states 1 and 2","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"transition_matrix(mc_map) - transition_matrix(mc_mle)","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"Finally, if we fear very small transition probabilities, we can perform the entire estimation in log scale thanks to LogarithmicNumbers.jl.","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"mc_mle_log = fit_mle(DiscreteMarkovChain{LogFloat32,LogFloat32}, state_sequence)\n\nerror_mle_log = mean(abs, transition_matrix(mc_mle_log) - transition_matrix(mc))","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"","category":"page"},{"location":"examples/discrete_markov/","page":"Discrete Markov chain","title":"Discrete Markov chain","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = HiddenMarkovModels","category":"page"},{"location":"#HiddenMarkovModels.jl","page":"Home","title":"HiddenMarkovModels.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Welcome to the documentation of HiddenMarkovModels.jl, a lightweight package for working with Markov chains and Hidden Markov Models.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install the package, open a Julia Pkg REPL and run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/gdalle/HiddenMarkovModels.jl","category":"page"},{"location":"#Mathematical-background","page":"Home","title":"Mathematical background","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To understand the algorithms implemented here, check out the following literature:","category":"page"},{"location":"","page":"Home","title":"Home","text":"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, Lawrence R. Rabiner (1989)","category":"page"},{"location":"#Alternatives","page":"Home","title":"Alternatives","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you don't find what you are looking for around here, there are several other Julia packages with a focus on Markovian modeling. Here are the ones that I am aware of:","category":"page"},{"location":"","page":"Home","title":"Home","text":"HMMBase.jl\nMarkovModels.jl\nHiddenMarkovModels.jl\nMitosis.jl\nContinuousTimeMarkov.jl\nPiecewiseDeterministicMarkovProcesses.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"The reason I implemented my own was because I needed specific features that were not simultaneously available elsewhere (to the best of my knowledge):","category":"page"},{"location":"","page":"Home","title":"Home","text":"Compatibility with generic emissions that go beyond Distributions.jl objects\nNumerical stability thanks to log-scale computations\nDiscrete and continuous time versions (WIP)\nMAP estimation with priors (WIP)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"EditURL = \"https://github.com/gdalle/HiddenMarkovModels.jl/blob/main/test/examples/hmm.jl\"","category":"page"},{"location":"examples/hmm/#Hidden-Markov-Model","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"","category":"section"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"using Distributions\nusing HiddenMarkovModels\nusing LogarithmicNumbers\nusing Plots\nusing Statistics","category":"page"},{"location":"examples/hmm/#Construction","page":"Hidden Markov Model","title":"Construction","text":"","category":"section"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"A HiddenMarkovModel object is build by combining a transition structure (of type DiscreteMarkovChain) with a list of emission distributions.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"p0 = [0.3, 0.7]\nP = [0.9 0.1; 0.2 0.8]\ntransitions = DiscreteMarkovChain(p0, P)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"emission1 = Normal(0.4, 0.7)\nemission2 = Normal(-0.8, 0.3)\nemissions = [emission1, emission2]","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"hmm = HiddenMarkovModel(transitions, emissions)","category":"page"},{"location":"examples/hmm/#Simulation","page":"Hidden Markov Model","title":"Simulation","text":"","category":"section"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"The simulation utility returns both the sequence of states and the sequence of observations.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"state_sequence, obs_sequence = rand(hmm, 10)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"With the learning step in mind, we want to generate multiple observations sequences of various lengths.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"obs_sequences = [rand(hmm, rand(1000:2000))[2] for k in 1:5];\nnothing #hide","category":"page"},{"location":"examples/hmm/#Learning","page":"Hidden Markov Model","title":"Learning","text":"","category":"section"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"The Baum-Welch algorithm for estimating HMM parameters requires an initial guess, which we choose arbitrarily. Initial parameters can be created with reduced precision to speed up estimation.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"p0_init = rand_prob_vec(Float32, 2)\nP_init = rand_trans_mat(Float32, 2)\ntransitions_init = DiscreteMarkovChain(p0_init, P_init)\nemissions_init = [Normal(one(Float32)), Normal(-one(Float32))]\n\nhmm_init = HiddenMarkovModel(transitions_init, emissions_init)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"We can now apply the algorithm by setting a tolerance on the loglikelihood increase, as well as a maximum number of iterations.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"hmm_est, logL_evolution = baum_welch_multiple_sequences(\n    hmm_init, obs_sequences; max_iterations=100, tol=1e-5\n);\nnothing #hide","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"As we can see on the plot, each iteration increases the loglikelihood of the estimate: it is a fundamental property of the EM algorithm and its variants.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"plot(\n    logL_evolution;\n    title=\"Baum-Welch convergence (Normal emissions)\",\n    xlabel=\"Iteration\",\n    ylabel=\"Log-likelihood\",\n    label=nothing,\n    margin=5Plots.mm\n)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"To improve numerical stability, we can apply the algorithm directly in log scale thanks to LogarithmicNumbers.jl.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"p0_init_log = rand_prob_vec(LogFloat32, 2)\nP_init_log = rand_trans_mat(LogFloat32, 2)\ntransitions_init_log = DiscreteMarkovChain(p0_init_log, P_init_log)\nemissions_init_log = [Normal(one(LogFloat32)), Normal(-one(LogFloat32))]\n\nhmm_init_log = HiddenMarkovModel(transitions_init_log, emissions_init_log)\n\nhmm_est_log, logL_evolution_log = baum_welch_multiple_sequences(\n    hmm_init_log, obs_sequences; max_iterations=100, tol=1e-5\n);\n\nplot(\n    logL_evolution_log;\n    title=\"Log Baum-Welch convergence (Normal emissions)\",\n    xlabel=\"Iteration\",\n    ylabel=\"Log-likelihood\",\n    label=nothing,\n    margin=5Plots.mm\n)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"Let us now compute the estimation error on various parameters.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"transition_error_init = mean(abs, transition_matrix(hmm_init) - transition_matrix(hmm))\nμ_error_init = mean(abs, [get_emission(hmm_init, s).μ - get_emission(hmm, s).μ for s in 1:2])\nσ_error_init = mean(abs, [get_emission(hmm_init, s).σ - get_emission(hmm, s).σ for s in 1:2])\n(transition_error_init, μ_error_init, σ_error_init)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"transition_error = mean(abs, transition_matrix(hmm_est) - transition_matrix(hmm))\nμ_error = mean(abs, [get_emission(hmm_est, s).μ - get_emission(hmm, s).μ for s in 1:2])\nσ_error = mean(abs, [get_emission(hmm_est, s).σ - get_emission(hmm, s).σ for s in 1:2])\n(transition_error, μ_error, σ_error)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"As we can see, all of these errors are much smaller than those of hmm_init: mission accomplished! The same goes for the logarithmic version.","category":"page"},{"location":"examples/hmm/#Custom-emission-distributions","page":"Hidden Markov Model","title":"Custom emission distributions","text":"","category":"section"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"One of the major selling points for HiddenMarkovModels.jl is that the user can define their own emission distributions. Here we give an example where emissions are of type MultivariatePoissonProcess with state-dependent rates.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"emissions_poisson = [\n    MultivariatePoissonProcess([1.0, 2.0, 3.0]), MultivariatePoissonProcess([3.0, 2.0, 1.0])\n]\n\nhmm_poisson = HMM(transitions, emissions_poisson)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"We can simulate and learn it using the exact same procedure.","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"state_sequence_poisson, obs_sequence_poisson = rand(hmm_poisson, 1000);\n\nemissions_init_poisson = [\n    MultivariatePoissonProcess([rand(), 2rand(), 3rand()]),\n    MultivariatePoissonProcess([3rand(), 2rand(), rand()]),\n]\n\nhmm_init_poisson = HMM(transitions_init, emissions_init_poisson)\n\nhmm_est_poisson, logL_evolution_poisson = baum_welch(\n    hmm_init_poisson, obs_sequence_poisson; max_iterations=100, tol=1e-5\n);\n\nplot(\n    logL_evolution_poisson;\n    title=\"Baum-Welch convergence (Poisson emissions)\",\n    xlabel=\"Iteration\",\n    ylabel=\"Log-likelihood\",\n    label=nothing,\n    margin=5Plots.mm\n)","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"","category":"page"},{"location":"examples/hmm/","page":"Hidden Markov Model","title":"Hidden Markov Model","text":"This page was generated using Literate.jl.","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"EditURL = \"https://github.com/gdalle/HiddenMarkovModels.jl/blob/main/test/examples/multivariate_poisson.jl\"","category":"page"},{"location":"examples/multivariate_poisson/#Multivariate-Poisson-process","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"","category":"section"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"using HiddenMarkovModels\nusing LogarithmicNumbers\nusing Plots\nusing Statistics","category":"page"},{"location":"examples/multivariate_poisson/#Construction","page":"Multivariate Poisson process","title":"Construction","text":"","category":"section"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"A MultivariatePoissonProcess object is built from a vector of positive event rates.","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"λ = rand(5)\npp = MultivariatePoissonProcess(λ)","category":"page"},{"location":"examples/multivariate_poisson/#Simulation","page":"Multivariate Poisson process","title":"Simulation","text":"","category":"section"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"Since it is a temporal point process, we can simulate it on an arbitrary real interval.","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"history = rand(pp, 3.14, 314.0)","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"Each event is defined by a time and an integer mark, which means we can visualize the history in 2 dimensions:","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"scatter(\n    event_times(history),\n    event_marks(history);\n    title=\"Event history\",\n    xlabel=\"Time\",\n    ylabel=\"Mark\",\n    label=nothing,\n    margin=5Plots.mm\n)","category":"page"},{"location":"examples/multivariate_poisson/#Learning","page":"Multivariate Poisson process","title":"Learning","text":"","category":"section"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"Parameters can learned with Maximum Likelihood Estimation (MLE):","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"pp_est = fit_mle(MultivariatePoissonProcess{Float32}, history)","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"Let's see how well we did","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"error = mean(abs, pp_est.λ - pp.λ)","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"","category":"page"},{"location":"examples/multivariate_poisson/","page":"Multivariate Poisson process","title":"Multivariate Poisson process","text":"This page was generated using Literate.jl.","category":"page"}]
}
