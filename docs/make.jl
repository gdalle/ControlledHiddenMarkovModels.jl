using Documenter
using ControlledHiddenMarkovModels
using Literate

DocMeta.setdocmeta!(
    ControlledHiddenMarkovModels,
    :DocTestSetup,
    :(using ControlledHiddenMarkovModels);
    recursive=true,
)

EXAMPLES_DIR_JL = joinpath(@__DIR__, "..", "test", "examples")
EXAMPLES_DIR_MD = joinpath(@__DIR__, "src", "examples")

# for file in readdir(EXAMPLES_DIR_MD)
#     if endswith(file, ".md")
#         rm(joinpath(EXAMPLES_DIR_MD, file))
#     end
# end

# for file in readdir(EXAMPLES_DIR_JL)
#     Literate.markdown(
#         joinpath(EXAMPLES_DIR_JL, file),
#         EXAMPLES_DIR_MD;
#         documenter=true,
#         flavor=Literate.DocumenterFlavor(),
#     )
# end

makedocs(;
    modules=[ControlledHiddenMarkovModels],
    authors="Guillaume Dalle <22795598+gdalle@users.noreply.github.com>",
    repo="https://github.com/gdalle/ControlledHiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="ControlledHiddenMarkovModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/ControlledHiddenMarkovModels.jl",
        assets=String[],
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        # "Examples" => [
        #     "Vanilla HMM" => "examples/hmm.md",
        #     "Controlled HMM" => "examples/hmm_controlled.md",
        # ],
        "API reference" => "api.md",
    ],
)

# for file in readdir(EXAMPLES_DIR_MD)
#     if endswith(file, ".md")
#         rm(joinpath(EXAMPLES_DIR_MD, file))
#     end
# end

deploydocs(; repo="github.com/gdalle/ControlledHiddenMarkovModels.jl", devbranch="main")
