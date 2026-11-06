# Discovery of the MF6 input files of a simulation, for the tests that loop over the
# MF6 examples. Expects `ModflowGPU` to be loaded.

using TOML: TOML

"""
Root of the reference material (validation/fetch_reference.sh): `REFERENCE_DIR`, default
`<repo root>/reference`.
"""
const REFERENCE_DIR = get(ENV, "REFERENCE_DIR", normpath(joinpath(@__DIR__, "..", "reference")))

"""
    in_scope_examples(; tiers = 0:7) -> Vector{Pair{String, String}}

`name => simulation directory` of the in-scope examples of `validation/examples.toml` in
the given tiers, sorted by name. Empty when the extracted examples are missing.
"""
function in_scope_examples(; tiers = 0:7)
    exdir = joinpath(REFERENCE_DIR, "mf6examples")
    isdir(exdir) || return Pair{String, String}[]
    toml = TOML.parsefile(joinpath(@__DIR__, "..", "validation", "examples.toml"))
    out = Pair{String, String}[]
    for (name, ex) in toml["examples"]
        ex["tier"] in tiers && push!(out, name => joinpath(exdir, ex["path"]))
    end
    return sort!(out)
end

"""
    simulation_input_files(simdir) -> Vector{ModflowGPU.InputFile}

Read every block-structured input file of the simulation in `simdir`: `mfsim.nam`, the
TDIS file, the model name files and their packages, exchanges, solutions, and every file
named after `FILEIN` in a file read before (TS6, TAS6, OBS6, ...), each checked against
its DFN definition.
"""
function simulation_input_files(simdir::AbstractString)
    M = ModflowGPU
    done = Dict{String, M.InputFile}()
    queue = Tuple{String, String, String}[]  # (file name, ftype, component)
    function visit(fname, ftype, component)
        path = normpath(joinpath(simdir, fname))
        haskey(done, path) && return nothing
        f = M.read_input_file(path; ftype, component, simdir)
        done[path] = f
        # Files named after FILEIN (TS6 FILEIN x.ts, OBS6 FILEIN x.obs, ...).
        for b in f.blocks, l in b.lines
            for j in 2:(length(l) - 1)
                M.keyword(l, j) == "FILEIN" && push!(queue, (l[j + 1], M.keyword(l, j - 1), component))
            end
        end
        return f
    end
    sim = visit("mfsim.nam", "SIM", "SIM")
    for l in M.find_block(sim, "TIMING").lines
        M.keyword(l, 1) == "TDIS6" && visit(l[2], "TDIS6", "SIM")
    end
    for l in M.find_block(sim, "MODELS").lines
        mtype = M.keyword(l, 1)
        component = rstrip(mtype, '6')
        nam = visit(l[2], mtype, component)
        for p in M.find_block(nam, "PACKAGES").lines
            visit(p[2], M.keyword(p, 1), component)
        end
    end
    exg = M.find_block(sim, "EXCHANGES")
    isnothing(exg) || foreach(l -> visit(l[2], M.keyword(l, 1), "EXG"), exg.lines)
    for b in M.find_blocks(sim, "SOLUTIONGROUP"), l in b.lines
        M.keyword(l, 1) == "MXITER" && continue
        visit(l[2], M.keyword(l, 1), "SLN")
    end
    while !isempty(queue)
        visit(popfirst!(queue)...)
    end
    return collect(values(done))
end
