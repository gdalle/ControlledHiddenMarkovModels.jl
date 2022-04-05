using Documenter
using HiddenMarkovModels
using Literate

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

EXAMPLES_DIR_JL = joinpath(dirname(@__DIR__), "test")
EXAMPLES_DIR_MD = joinpath(@__DIR__, "src", "examples")

for file in readdir(EXAMPLES_DIR_MD)
    if endswith(file, ".md")
        rm(joinpath(EXAMPLES_DIR_MD, file))
    end
end

for file in readdir(EXAMPLES_DIR_JL)
    if endswith(file, ".jl") && !startswith(file, "runtests")
        Literate.markdown(
            joinpath(EXAMPLES_DIR_JL, file),
            EXAMPLES_DIR_MD;
            documenter=true,
            flavor=Literate.DocumenterFlavor(),
        )
    end
end

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle <22795598+gdalle@users.noreply.github.com> and contributors",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Discrete Markov chain" => "examples/discrete_markov.md",
            "Hidden Markov Model" => "examples/hmm.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
