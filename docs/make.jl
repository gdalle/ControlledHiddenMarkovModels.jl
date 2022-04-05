using HiddenMarkovModels
using Documenter

DocMeta.setdocmeta!(HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true)

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
    ],
)

deploydocs(;
    repo="github.com/gdalle/HiddenMarkovModels.jl",
    devbranch="main",
)
