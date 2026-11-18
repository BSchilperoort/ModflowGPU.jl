# The parity table of the manual (docs/src/assets/parity.toml, written by
# docs/parity/collect.jl, T8.4) has a row for every model of the run and synthetic parity
# tests (test/parity/run_test.jl, test/parity/synthetic_test.jl), and every row passed.
# Without reference/ (CI) only the test models and the synthetic models are checked.
using TOML: TOML

@testset "Parity table of the manual" begin
    toml = TOML.parsefile(joinpath(pkgdir(ModflowGPU), "docs", "src", "assets", "parity.toml"))
    rows = Dict(r["name"] => r for r in toml["examples"])
    names = [first.(run_parity_models()); ["synthetic $s" for s in SYNTHETIC_PARITY_MODELS]]
    missing_rows = filter(n -> !haskey(rows, n), names)
    isempty(missing_rows) || @info "Not in docs/src/assets/parity.toml (run docs/parity/collect.jl): $(join(missing_rows, ", "))"
    @test isempty(missing_rows)
    @test all(r -> r["passed"], values(rows))
    @test all(r -> issubset(["cpu"], keys(r["results"])), values(rows))
end
