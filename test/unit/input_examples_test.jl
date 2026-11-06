# Every input file of every in-scope MF6 example (validation/examples.toml, tiers 0-7)
# tokenises into blocks that match its DFN definition (T1.1 done criterion).
@testset "MF6 examples tokenise into blocks" begin
    examples = in_scope_examples(; tiers = 0:7)
    if isempty(examples)
        @info "Skipping the MF6 example input test: $(joinpath(REFERENCE_DIR, "mf6examples")) not found (validation/fetch_reference.sh)"
    else
        nfiles = 0
        nblocks = 0
        for (name, simdir) in examples
            @testset "$name" begin
                files = simulation_input_files(simdir)
                @test !isempty(files)
                @test all(f -> !isnothing(f.dfn), files)
                nfiles += length(files)
                nblocks += sum(f -> length(f.blocks), files)
            end
        end
        @info "MF6 examples: $(length(examples)) simulations, $nfiles input files, $nblocks blocks"
    end
end
