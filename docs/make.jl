# Build the user manual: `julia --project=docs docs/make.jl`
# Set LINKCHECK=true to check external links (CI does; it needs network access).
# deploydocs does nothing outside CI.
using Documenter
using ModflowGPU

# Tables of the MODFLOW 6 compatibility pages (Main.CompatTables, used by their @eval blocks).
include(joinpath(@__DIR__, "compat_tables.jl"))

# Figures of the Performance page, from docs/src/assets/performance.toml (checked against
# benchmark/results/); written to docs/src/assets/ (gitignored).
include(joinpath(@__DIR__, "performance_figures.jl"))
PerformanceFigures.make_figures(joinpath(@__DIR__, "src", "assets"))

DocMeta.setdocmeta!(ModflowGPU, :DocTestSetup, :(using ModflowGPU); recursive = true)

makedocs(;
    sitename = "ModflowGPU.jl",
    modules = [ModflowGPU],
    authors = "Bart Schilperoort",
    repo = Documenter.Remotes.GitHub("BSchilperoort", "ModflowGPU.jl"),
    format = Documenter.HTML(;
        canonical = "https://bschilperoort.github.io/ModflowGPU.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "install.md",
        "Getting started" => "getting_started.md",
        "GPU guide" => "gpu.md",
        "MODFLOW 6 compatibility" => [
            "Supported packages" => "compatibility/packages.md",
            "Comparing results" => "compatibility/results.md",
            "Differences from MODFLOW 6" => "compatibility/differences.md",
        ],
        "Coupling" => [
            "Stepping API" => "coupling/stepping.md",
            "BMI" => "coupling/bmi.md",
        ],
        "Performance" => "performance.md",
        "API reference" => [
            "Overview" => "api/index.md",
            "Running a simulation" => "api/running.md",
            "Backends and precision" => "api/backends.md",
            "Stepping API" => "api/stepping.md",
            "Building models" => "api/builder.md",
            "Output files" => "api/output.md",
            "For backend authors" => "api/backend_authors.md",
        ],
        "For developers" => "developers.md",
    ],
    # Warnings are errors (no `warnonly`): broken cross-references, doctest failures and
    # missing docstrings fail the build.
    doctest = true,
    linkcheck = get(ENV, "LINKCHECK", "false") == "true",
    # doi.org refuses bots; GitHub rate-limits (429) when many links go to the same repo, and
    # links to files of this repository only resolve once they are pushed.
    linkcheck_ignore = [r"^https://doi\.org/", r"^https://github\.com/BSchilperoort/ModflowGPU\.jl/"],
    # Every exported or `public` name must be in an `@docs` block (docs/src/api/); internal
    # names are documented in docs/dev only. test/unit/api_test.jl checks the same list.
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/BSchilperoort/ModflowGPU.jl.git",
    devbranch = "main",
    push_preview = false,
)
