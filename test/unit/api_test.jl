# Public API (T8.2): the exported and `public` names of ModflowGPU match the names listed
# in the `@docs` blocks of the API pages (docs/src/api/*.md), each on exactly one page,
# and every one of them has a docstring.

# Names listed in the `@docs` blocks of a Markdown page, without the `ModflowGPU.` prefix.
function _docs_block_names(path)
    names = Symbol[]
    inblock = false
    for line in eachline(path)
        s = strip(line)
        if !inblock
            inblock = s == "```@docs"
        elseif s == "```"
            inblock = false
        elseif !isempty(s)
            push!(names, Symbol(replace(s, r"^ModflowGPU\." => "")))
        end
    end
    return names
end

@testset "Public API" begin
    public_names = Set(n for n in names(ModflowGPU; all = true) if Base.ispublic(ModflowGPU, n))
    exported = Set(n for n in public_names if Base.isexported(ModflowGPU, n))

    @testset "API pages list every public name once" begin
        apidir = joinpath(pkgdir(ModflowGPU), "docs", "src", "api")
        pages = filter(endswith(".md"), readdir(apidir; join = true))
        listed = Dict(basename(p) => _docs_block_names(p) for p in pages)
        all_listed = reduce(vcat, values(listed))
        counts = Dict(n => count(==(n), all_listed) for n in unique(all_listed))
        @test isempty(setdiff(public_names, keys(counts)))   # public but on no page
        @test isempty(setdiff(keys(counts), public_names))   # on a page but not public
        @test isempty([n for (n, c) in counts if c > 1])     # on more than one page
    end

    @testset "Docstrings" begin
        @test isempty([n for n in public_names if !Base.Docs.hasdoc(ModflowGPU, n)])
    end

    @testset "No clash with Base" begin
        # Exported names must not shadow Base's exports (decision 7: run_simulation and
        # finalize_simulation instead of run and finalize).
        @test isempty([n for n in exported if n !== :ModflowGPU && Base.isexported(Base, n)])
        @test :run_simulation in exported && :finalize_simulation in exported
        @test !(:run in exported) && !(:finalize in exported) && !(:solve in exported)
    end

    @testset "Aliases" begin
        @test ModflowGPU.run === run_simulation
        @test ModflowGPU.finalize === finalize_simulation
        @test Base.ispublic(ModflowGPU, :run) && Base.ispublic(ModflowGPU, :finalize)
        # `using ModflowGPU` leaves Base's functions alone.
        @test run === Base.run
        @test finalize === Base.finalize
    end
end
