# READARRAYGRID input (T6.18, src/input/lists.jl `read_arraygrid_periods`) and IMS
# SCALING_METHOD L2NORM:
#
# 1. The synthetic model test/models/arraygrid (CHDG, WELG, DRNG, RIVG, GHBG): the records
#    MF6 builds (Mf6FileGridArray.f90 `param_load`: the first array of a block defines the
#    records, later arrays are taken at those cells, arrays not given keep their values
#    record by record; `nodeu_to_nlist` drops the inactive last GHB cell), errors, and run
#    parity against mf6 6.8.2 run at test time (`_run_parity`, test/run_parity.jl).
# 2. ex-gwf-twri01 and ex-gwf-bcf2ss-p01a with their WEL, DRN, RIV and CHD lists rewritten
#    as READARRAYGRID (`_arraygrid_variant`): the records equal the list records sorted by
#    node, our heads equal those of the list form bitwise (`:single`, `:double`), and the
#    variant is within run parity of mf6.
# 3. IMS SCALING_METHOD L2NORM (ImsLinearBase.f90 `ims_base_scale`, T2.7), accepted with
#    `strict = true` since T6.18: run parity of test/models/coupling_rde and
#    test/models/arraygrid with L2NORM scaling (BICGSTAB) against mf6 run at test time.
using ModflowGPU: load_simulation, read_input_file, read_period_data, read_arraygrid_periods, grid_dims, dis_dims,
    boundary_kind, DNODATA, MF6UnsupportedError, MF6InputError

const AG_MODEL = joinpath(@__DIR__, "..", "models", "arraygrid")

function _ag_unsupported(f)
    try
        f()
    catch e
        e isa MF6UnsupportedError && return e
        rethrow()
    end
    @test false
    return nothing
end

_ag_boundary(sim, kind) = only(b for b in only(sim.models).boundaries if boundary_kind(b) === kind)

# A copy of `src` (simulation inputs only) with `edit(name, text)` applied to every file.
function _ag_copy(src, dst; edit = (f, t) -> t)
    mkpath(dst)
    for f in readdir(src)
        any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".txt", ".toml", ".grb", ".csv")) && !startswith(f, "ag_") && continue
        isdir(joinpath(src, f)) && continue
        write(joinpath(dst, f), edit(f, read(joinpath(src, f), String)))
    end
    return dst
end

# `src` with every WEL, DRN, RIV, GHB and CHD list file rewritten as READARRAYGRID input:
# the OPTIONS and DIMENSIONS blocks are kept (READARRAYGRID added), each PERIOD block gives
# one INTERNAL array of NODES values per column and auxiliary variable (DNODATA where there
# is no record). Needs lists without time series, boundnames or two records in a cell.
function _arraygrid_variant(src, dst)
    _ag_copy(src, dst)
    m = only(load_simulation(src).models)
    nodes = prod(grid_dims(m.dis).mshape)
    for b in m.boundaries
        boundary_kind(b) in (:WEL, :DRN, :RIV, :GHB, :CHD) || continue
        @assert !b.boundnames && isempty(b.ts_files)
        path = joinpath(dst, b.file)
        txt = read(path, String)
        i = findfirst(r"^\s*BEGIN\s+period"mi, txt)
        io = IOBuffer()
        print(io, replace(txt[1:(first(i) - 1)], r"^(\s*BEGIN\s+options[^\n]*\n)"mi => s"\1  READARRAYGRID\n"))
        cols = [(uppercase(c.name), c.first, false) for c in b.schema.columns if c.kind === :double]
        append!(cols, [(a, k, true) for (k, a) in enumerate(b.auxnames)])
        for (iper, p) in zip(b.periods.iper, b.periods.data)
            @assert allunique(p.nodeu)
            println(io, "BEGIN period  $iper")
            for (name, row, aux) in cols
                vals = aux ? p.auxvar[row, :] : p.bound[row, :]
                @assert !any(==(DNODATA), vals)
                a = fill(DNODATA, nodes)
                a[p.nodeu] .= vals
                println(io, "  $name\n    INTERNAL")
                for part in Iterators.partition(a, 10)
                    println(io, "      ", join(repr.(part), "  "))
                end
            end
            println(io, "END period  $iper\n")
        end
        write(path, String(take!(io)))
    end
    return dst
end

# Run mf6 in `dir`; false (logged) when mf6 is missing.
function _ag_mf6(dir)
    if !isfile(RUN_MF6)
        @info "Skipping the mf6 run in $(basename(dir)): mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
        return false
    end
    out = read(Cmd(`$RUN_MF6`; dir), String)
    @test occursin("Normal termination", out)
    return true
end

@testset "READARRAYGRID: records of test/models/arraygrid" begin
    sim = load_simulation(AG_MODEL)  # strict
    wel = _ag_boundary(sim, :WEL)
    @test wel.maxbound == 40 && wel.auxmultcol == 1 && wel.periods.iper == [1, 2, 3, 4]
    p = wel.periods.data
    @test p[1].nodeu == [8, 14, 33] && p[1].bound == [-50.0 -30.0 -40.0] && p[1].auxvar == [1.0 1.0 1.0]
    # Only MULT given: Q keeps its values.
    @test p[2].nodeu == [8, 14, 33] && p[2].bound == [-50.0 -30.0 -40.0] && p[2].auxvar == [0.5 2.0 1.0]
    # MULT defines four records; Q keeps its values record by record (zero beyond them).
    @test p[3].nodeu == [8, 14, 20, 33] && p[3].bound == [-50.0 -30.0 -40.0 0.0]
    @test isempty(p[4].nodeu) && size(p[4].bound) == (1, 0)
    drn = _ag_boundary(sim, :DRN)
    @test drn.periods.data[1].nodeu == [2, 3] && drn.periods.data[1].bound == [9.5 9.4; 100.0 100.0]
    riv = _ag_boundary(sim, :RIV)
    @test riv.periods.iper == [1, 3]
    @test riv.periods.data[1].nodeu == [9, 19] && riv.periods.data[1].bound == [9.0 8.8; 200.0 200.0; 8.0 7.9]
    @test riv.periods.data[2].bound == [9.2 9.0; 200.0 200.0; 8.0 7.9]
    # The third GHB cell is inactive: dropped.
    ghb = _ag_boundary(sim, :GHB)
    @test ghb.maxbound == 3 && ghb.periods.data[1].nodeu == [25, 30] && ghb.periods.data[1].bound == [7.0 7.5; 50.0 50.0]
    chd = _ag_boundary(sim, :CHD)
    @test chd.periods.data[1].nodeu == [1, 6, 11, 16] && chd.periods.data[1].bound == fill(10.0, 1, 4)

    mktempdir() do dir
        g = dis_dims(2, 4, 5)
        pkg(name, text) = read_input_file(write_input(dir, name, text); dfn = "gwf-chd", simdir = dir)
        opts = "BEGIN OPTIONS\n READARRAYGRID\nEND OPTIONS\n"
        # read_period_data dispatches on the variant; the schema is that of the list form.
        s, sp = read_period_data(pkg("a.chdg", opts * "BEGIN PERIOD 2\n HEAD\n CONSTANT 1.5\nEND PERIOD\n"), g)
        @test s.package == "gwf-chd" && s.maxbound == 40 && sp.iper == [2] && sp.data[1].nodeu == 1:40
        # More defined cells than MAXBOUND.
        f = pkg("b.chdg", opts * "BEGIN DIMENSIONS\n MAXBOUND 2\nEND DIMENSIONS\nBEGIN PERIOD 1\n HEAD\n CONSTANT 1.0\nEND PERIOD\n")
        @test occursin("exceeds MAXBOUND=2", input_error_of(() -> read_arraygrid_periods(f, g)).msg)
        # Unknown array, NETCDF, TIMEARRAYSERIES.
        f = pkg("c.chdg", opts * "BEGIN PERIOD 1\n STAGE\n CONSTANT 1.0\nEND PERIOD\n")
        @test occursin("\"STAGE\" is not a PERIOD array", input_error_of(() -> read_arraygrid_periods(f, g)).msg)
        f = pkg("d.chdg", opts * "BEGIN PERIOD 1\n HEAD NETCDF\nEND PERIOD\n")
        @test occursin("NETCDF", input_error_of(() -> read_arraygrid_periods(f, g)).msg)
        f = pkg("e.chdg", opts * "BEGIN PERIOD 1\n HEAD TIMEARRAYSERIES h\nEND PERIOD\n")
        @test occursin("TIMEARRAYSERIES", input_error_of(() -> read_arraygrid_periods(f, g)).msg)
        # All values DNODATA: no records; the next array then defines them.
        f = pkg("f.chdg", opts * "BEGIN PERIOD 1\n HEAD\n CONSTANT 3.0e30\nEND PERIOD\n")
        @test isempty(read_arraygrid_periods(f, g)[2].data[1].nodeu)
        f = pkg("g.chdg", "BEGIN OPTIONS\n READARRAYGRID\n AUX A\nEND OPTIONS\nBEGIN PERIOD 1\n HEAD\n CONSTANT 3.0e30\n A\n CONSTANT 2.0\n HEAD\n CONSTANT 4.0\nEND PERIOD\n")
        q = read_arraygrid_periods(f, g)[2].data[1]
        @test q.nodeu == 1:40 && all(==(4.0), q.bound) && all(==(2.0), q.auxvar)
    end

    # Strict loading: EXPORT_ARRAY_NETCDF (and other options outside the table) unsupported.
    mktempdir() do tmp
        dir = _ag_copy(AG_MODEL, joinpath(tmp, "nc"); edit = (f, t) -> f == "ag.chdg" ? replace(t, "SAVE_FLOWS" => "SAVE_FLOWS\n  EXPORT_ARRAY_NETCDF") : t)
        @test occursin("CHD6 option EXPORT_ARRAY_NETCDF", _ag_unsupported(() -> load_simulation(dir)).msg)
        @test load_simulation(dir; strict = false) isa ModflowGPU.SimulationSpec
    end
end

@testset "READARRAYGRID: run parity of test/models/arraygrid vs MF6" begin
    mktempdir() do tmp
        dir = _ag_copy(AG_MODEL, joinpath(tmp, "arraygrid"))
        _ag_mf6(dir) && _run_parity("arraygrid", dir)
    end
end

@testset "READARRAYGRID: $ex rewritten from list input" for ex in ("ex-gwf-twri01", "ex-gwf-bcf2ss-p01a")
    src = reference_rundir(ex)
    if isnothing(src)
        @info "Skipping the READARRAYGRID variant of $ex: no reference run in $(joinpath(REFERENCE_DIR, "runs", ex))"
        continue
    end
    mktempdir() do tmp
        dir = _arraygrid_variant(src, joinpath(tmp, ex))
        list = only(load_simulation(src).models)
        grid = only(load_simulation(dir).models)
        n = 0
        for (bl, bg) in zip(list.boundaries, grid.boundaries)
            @test boundary_kind(bl) === boundary_kind(bg)
            boundary_kind(bl) in (:WEL, :DRN, :RIV, :GHB, :CHD) || continue
            n += 1
            @test bg.periods.iper == bl.periods.iper
            for (pl, pg) in zip(bl.periods.data, bg.periods.data)
                perm = sortperm(pl.nodeu)
                @test pg.nodeu == pl.nodeu[perm] && pg.bound == pl.bound[:, perm] && pg.auxvar == pl.auxvar[:, perm]
            end
        end
        @test n >= 2
        # Our runs of the list and grid forms: the same heads, bit for bit.
        for_each_precision() do p
            rl = ModflowGPU.run(src; backend = TEST_BACKEND, precision = p.preset, write_output = false)
            rg = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, write_output = false)
            @test rl.converged && rg.converged
            @test rg.heads == rl.heads
        end
        _ag_mf6(dir) && _run_parity("$ex (READARRAYGRID)", dir)
    end
end

@testset "IMS SCALING_METHOD L2NORM: run parity vs MF6" begin
    # test/models/arraygrid: with L2NORM scaling MF6's inner residual stalls near 5e-8 in
    # step 2 of period 2 (INNER_RCLOSE 1e-10 is never met and the solution fails), so its
    # variant has INNER_RCLOSE 1e-7 (the heads are converged by INNER_DVCLOSE 1e-11).
    l2(f, t, rclose) = endswith(f, ".ims") ?
        replace(t, r"^(\s*LINEAR_ACCELERATION\s+BICGSTAB\s*)$"mi => s"\1\n  SCALING_METHOD  L2NORM", r"INNER_RCLOSE\s+\S+" => "INNER_RCLOSE  $rclose") : t
    for (name, src, rclose) in (("coupling_rde", joinpath(@__DIR__, "..", "models", "coupling_rde"), "1.0e-10"), ("arraygrid", AG_MODEL, "1.0e-7"))
        mktempdir() do tmp
            dir = _ag_copy(src, joinpath(tmp, name); edit = (f, t) -> l2(f, t, rclose))
            sim = load_simulation(dir)  # strict: accepted since T6.18
            @test only(only(sim.solution_groups).solutions).scaling_method === :l2norm
            _ag_mf6(dir) && _run_parity("$name (L2NORM)", dir)
        end
    end
end

# T6.18 item 3: REORDERING_METHOD RCM and MD only change MF6's ILU order; they are accepted
# with an Info and change nothing in our run.
@testset "IMS REORDERING_METHOD RCM and MD: accepted, no effect" begin
    ref = ModflowGPU.run(AG_MODEL; backend = TEST_BACKEND, precision = :double, write_output = false)
    for m in ("RCM", "MD")
        mktempdir() do tmp
            dir = _ag_copy(AG_MODEL, joinpath(tmp, m); edit = (f, t) -> endswith(f, ".ims") ? replace(t, "END linear" => "  REORDERING_METHOD  $m\nEND linear") : t)
            @test only(only(load_simulation(dir).solution_groups).solutions).reordering_method === Symbol(lowercase(m))
            r = @test_logs (:info, Regex("REORDERING_METHOD $m has no effect")) match_mode = :any ModflowGPU.run(
                dir; backend = TEST_BACKEND, precision = :double, write_output = false,
            )
            @test r.heads == ref.heads
        end
    end
end
