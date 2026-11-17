# Parity of the synthetic coupling models (T3.4) against mf6 6.8.2: `tiny` (50 × 40) and
# `indus` (204 × 180, the Indus grid) of benchmark/synthetic.jl, written with
# `write_simulation` to a temporary directory and run with mf6 at test time; steady state
# plus 12 daily transient periods (RCH and RIV stage change every period). Heads, CBC and
# listing budgets per PLAN.md section 4 through `_run_parity` (test/run_parity.jl), in
# both presets and, on a GPU backend, also against the CPU backend. Skips without mf6.
# MFGPU_TEST_SYNTHETIC=tiny runs only `tiny` (default "tiny,indus").
isdefined(@__MODULE__, :SyntheticModels) || include(joinpath(@__DIR__, "..", "..", "benchmark", "synthetic.jl"))

@testset "Synthetic model parity vs MF6 (T3.4)" begin
    if !isfile(RUN_MF6)
        @info "Skipping the synthetic model parity test: mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
    else
        for preset in split(get(ENV, "MFGPU_TEST_SYNTHETIC", join(SYNTHETIC_PARITY_MODELS, ',')), ',')
            @testset "$preset" begin
                mktempdir() do dir
                    SyntheticModels.write_synthetic(dir, Symbol(preset))
                    t = @elapsed out = read(Cmd(`$RUN_MF6`; dir), String)
                    @test occursin("Normal termination", out)
                    @info "synthetic $preset: mf6 6.8.2 ran in $(round(t; digits = 2)) s"
                    _run_parity("synthetic $preset", dir)
                end
            end
        end
    end
end
