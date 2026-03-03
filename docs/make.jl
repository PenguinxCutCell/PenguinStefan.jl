using Documenter
using PenguinStefan

makedocs(
    modules = [PenguinStefan],
    authors = "PenguinxCutCell contributors",
    sitename = "PenguinStefan.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/PenguinStefan.jl",
        repolink = "https://github.com/PenguinxCutCell/PenguinStefan.jl",
        collapselevel = 2,
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md",
        "Algorithms" => "algorithms.md",
        "Stefan Models" => "stefan.md",
    ],
    pagesonly = true,
    warnonly = false,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/PenguinStefan.jl",
        push_preview = true,
    )
end
