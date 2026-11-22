# CSV output (T6.17): OC BUDGETCSV, IMS CSV_OUTER_OUTPUT and CSV_INNER_OUTPUT, WEL
# AUTO_FLOW_REDUCE_CSV, against mf6 6.8.2 run at test time on copies of ex-gwf-fhb (CG),
# test/models/afr (BiCGSTAB, AUTO_FLOW_REDUCE) and ex-gwf-u1gwfgwf-s1 (two models): the
# headers equal MF6's; the budget and WEL values match within the budget tolerances of the
# run parity test (1e-6 of the total inflow in `:double`, 1e-4 in `:single`); the outer CSV
# matches in `:double` (the same outer iterations, `hncg` within 1e-7, its model and node);
# the inner CSV is checked for its format and its consistency with the outer CSV and the
# step results (inner iteration counts differ from MF6's by design, differences.md).

using ModflowGPU

# Add option lines to the OPTIONS block of `file` (a block is created when missing).
function _csv_add_options(file, lines...)
    t = read(file, String)
    l = join(lines, "\n  ")
    t = occursin(r"BEGIN\s+options"i, t) ? replace(t, r"BEGIN\s+options"i => "BEGIN options\n  $l"; count = 1) : "BEGIN options\n  $l\nEND options\n\n" * t
    write(file, t)
    return file
end

const _CSV_IMS_LINES = ("CSV_OUTER_OUTPUT FILEOUT sln.outer.csv", "CSV_INNER_OUTPUT FILEOUT sln.inner.csv")

function _fhb_with_csv(dir, src)
    cp(src, dir)
    _csv_add_options(joinpath(dir, "ex-gwf-fhb.ims"), _CSV_IMS_LINES...)
    _csv_add_options(joinpath(dir, "ex-gwf-fhb.oc"), "BUDGETCSV FILEOUT fhb.bud.csv")
    return dir, ["sln.outer.csv", "sln.inner.csv", "fhb.bud.csv"]
end

function _afr_with_csv(dir; ims = true)
    cp(joinpath(@__DIR__, "..", "models", "afr"), dir)
    ims && _csv_add_options(joinpath(dir, "afr.ims"), _CSV_IMS_LINES...)
    _csv_add_options(joinpath(dir, "afr.oc"), "BUDGETCSV FILEOUT afr.bud.csv")
    _csv_add_options(joinpath(dir, "afr.wel"), "AUTO_FLOW_REDUCE_CSV FILEOUT afr.frac.csv")
    _csv_add_options(joinpath(dir, "afr_len.wel"), "AUTO_FLOW_REDUCE_CSV FILEOUT afr.len.csv")
    return dir, [(ims ? ["sln.outer.csv", "sln.inner.csv"] : String[]); "afr.bud.csv"; "afr.frac.csv"; "afr.len.csv"]
end

function _u1_with_csv(dir, src)
    cp(src, dir)
    _csv_add_options(joinpath(dir, "ex-gwf-u1gwfgwf-s1.ims"), _CSV_IMS_LINES...)
    for m in ("inner", "outer")
        _csv_add_options(joinpath(dir, "$m.oc"), "BUDGETCSV FILEOUT $m.bud.csv")
    end
    return dir, ["sln.outer.csv", "sln.inner.csv", "inner.bud.csv", "outer.bud.csv"]
end

_csv_rows(path) = [split(l, ',') for l in readlines(path)]
_csv_num(s) = parse(Float64, s)

# Budget CSV: max |Δ rate| per row relative to MF6's TOTAL_IN, and max |Δ PERCENT_DIFFERENCE|.
function _budget_csv_difference(ours, mf6)
    a, b = _csv_rows(ours), _csv_rows(mf6)
    @test a[1] == b[1]
    @test length(a) == length(b)
    dq = dpd = 0.0
    for (ra, rb) in zip(a[2:end], b[2:end])
        @test ra[1] == rb[1]
        x, y = _csv_num.(ra[2:end]), _csv_num.(rb[2:end])
        tin = max(y[end - 2], floatmin())
        dq = max(dq, maximum(abs.(x[1:(end - 1)] .- y[1:(end - 1)])) / tin)
        dpd = max(dpd, abs(x[end] - y[end]))
    end
    return dq, dpd
end

# AUTO_FLOW_REDUCE_CSV: the same rows (time, period, step, record, cell as MF6's text),
# max |Δ| of the rates relative to the requested rate.
function _afr_csv_difference(ours, mf6)
    a, b = _csv_rows(ours), _csv_rows(mf6)
    @test a[1] == b[1]
    @test length(a) == length(b) > 1
    d = 0.0
    for (ra, rb) in zip(a[2:end], b[2:end])
        @test ra[1:5] == rb[1:5]
        x, y = _csv_num.(ra[6:8]), _csv_num.(rb[6:8])
        d = max(d, maximum(abs.(x .- y)) / abs(y[1]))
    end
    return d
end

# Outer CSV in `:double`: the same outer iterations, `hncg` within `tol`, its model and node
# where it is larger than `tol` (`same_node(a, b)`: nodes that tie, e.g. by symmetry).
function _outer_csv_compare(ours, mf6; tol = 1.0e-7, same_node = ==)
    a, b = _csv_rows(ours), _csv_rows(mf6)
    @test a[1] == b[1]
    @test length(a) == length(b)
    length(a) == length(b) || return Inf
    d = 0.0
    for (ra, rb) in zip(a[2:end], b[2:end])
        @test ra[2:5] == rb[2:5]
        x, y = _csv_num(ra[7]), _csv_num(rb[7])
        d = max(d, abs(x - y))
        @test ra[9] == rb[9] == ""
        abs(y) > tol && @test ra[8] == rb[8] && same_node(parse(Int, ra[10]), parse(Int, rb[10]))
    end
    @test d <= tol
    return d
end

# Our outer and inner CSV against the step results of the run and each other.
function _csv_consistency(dir, r; nmodels = 1, bcgs = false, nodesuser)
    outer, inner = _csv_rows(joinpath(dir, "sln.outer.csv")), _csv_rows(joinpath(dir, "sln.inner.csv"))
    nout = sum(s -> s.outer.iterations, r.steps)
    ninn = sum(s -> s.outer.inner, r.steps)
    @test length(outer) == nout + 1
    @test length(inner) == ninn + 1
    @test parse(Int, outer[end][1]) == ninn
    @test all(row -> length(row) == length(inner[1]), inner)
    @test length(inner[1]) == 13 + bcgs + (nmodels > 1 ? 4 * nmodels : 0)
    @test [parse(Int, row[1]) for row in inner[2:end]] == 1:ninn
    # The inner rows of every outer iteration: its time, period, step and outer iteration,
    # inner iterations 1 to its count.
    i = 1
    tot = 0
    for row in outer[2:end]
        n = parse(Int, row[6])
        tot += n
        @test parse(Int, row[1]) == tot
        for k in 1:n
            @test inner[i + k][2:5] == row[2:5] && parse(Int, inner[i + k][6]) == k
        end
        i += n
    end
    for row in inner[2:end]
        for (v, m, u) in ((7, 8, 9), (10, 11, 12))
            x, mid, node = _csv_num(row[v]), parse(Int, row[m]), parse(Int, row[u])
            @test x == 0 ? (mid, node) == (0, 0) : (1 <= mid <= nmodels && 1 <= node <= nodesuser[mid])
        end
        @test all(isfinite ∘ _csv_num, row[13:(13 + bcgs)])
    end
    return nothing
end

@testset "CSV output vs mf6" begin
    if !isfile(RUN_MF6)
        @info "Skipping the CSV output runs: mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
    else
        cases = Any[("afr", tmp -> _afr_with_csv(joinpath(tmp, "afr")), false, true)]
        for (name, make, cg) in (("ex-gwf-fhb", _fhb_with_csv, true), ("ex-gwf-u1gwfgwf-s1", _u1_with_csv, false))
            src = reference_rundir(name)
            if isnothing(src)
                @info "Skipping $name in the CSV output test: no reference run in $(joinpath(REFERENCE_DIR, "runs", name))"
            else
                push!(cases, (name, tmp -> make(joinpath(tmp, name), src), cg, false))
            end
        end
        for (name, make, cg, afr) in cases
            mktempdir() do tmp
                rundir, files = make(tmp)
                out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
                @test occursin("Normal termination", out)
                sim = load_simulation(rundir)
                nodesuser = [Int(ModflowGPU.nodes_user(m.dis)) for m in sim.models]
                @testset "$name" begin
                    for_each_precision() do p
                        mktempdir() do ours
                            r = run_simulation(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = ours)
                            @test r.converged
                            @test sort(basename.(r.csv_files)) == sort(files)
                            btol = p.T === Float64 ? 1.0e-6 : 1.0e-4
                            for f in files
                                a, b = joinpath(ours, f), joinpath(rundir, f)
                                @test readline(a) == readline(b)
                                if endswith(f, ".bud.csv")
                                    dq, dpd = _budget_csv_difference(a, b)
                                    @info "Budget CSV vs mf6: $name $f, :$(p.preset): max |Δ rate| $(round(dq; sigdigits = 3)) × total in, |Δ percent difference| $(round(dpd; sigdigits = 3))"
                                    @test dq <= btol
                                    @test dpd <= 200 * btol
                                elseif startswith(f, "afr.")
                                    d = _afr_csv_difference(a, b)
                                    @info "AUTO_FLOW_REDUCE_CSV vs mf6: $f, :$(p.preset): max |Δ| $(round(d; sigdigits = 3)) × requested rate"
                                    @test d <= btol
                                elseif f == "sln.outer.csv" && p.T === Float64
                                    # ex-gwf-fhb (3 rows, 10 columns) is symmetric about its middle
                                    # row: the largest changes of rows 1 and 3 tie at the
                                    # rounding level, and either may be the one reported.
                                    mirror(n) = (n - 1) % 10 + 1 + 10 * (2 - (n - 1) ÷ 10)
                                    d = _outer_csv_compare(a, b; same_node = name == "ex-gwf-fhb" ? ((x, y) -> x == y || mirror(x) == y) : ==)
                                    @info "Outer CSV vs mf6: $name, :$(p.preset): max |Δ hncg| $(round(d; sigdigits = 3))"
                                end
                            end
                            _csv_consistency(ours, r; nmodels = length(sim.models), bcgs = !cg, nodesuser)
                        end
                    end
                end
            end
        end
    end
end

@testset "CSV output: no effect on the solution, stepping, write_output" begin
    mktempdir() do tmp
        rundir, files = _afr_with_csv(joinpath(tmp, "afr"))
        for_each_precision() do p
            # The inner trace reads the solver vectors only: heads and budgets bitwise those of
            # a run without CSV output.
            r0 = run_simulation(joinpath(@__DIR__, "..", "models", "afr"); backend = TEST_BACKEND, precision = p.preset, write_output = false)
            r = run_simulation(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = joinpath(tmp, "out_$(p.preset)"))
            @test isempty(r0.csv_files)
            @test r.heads == r0.heads
            @test [t.rate_in for b in r.budgets for t in b.terms] == [t.rate_in for b in r0.budgets for t in b.terms]
            @test [s.outer.inner for s in r.steps] == [s.outer.inner for s in r0.steps]
        end
        # Without write_output (and through the stepping API) the CSV files are written, a
        # row per outer iteration and per time step.
        out = joinpath(tmp, "stepping")
        sim = initialize(rundir; backend = TEST_BACKEND, precision = :double, write_output = false, outdir = out)
        ModflowGPU.update(sim)
        ModflowGPU.update(sim)
        finalize_simulation(sim)
        @test sort(readdir(out)) == sort(files)
        @test length(readlines(joinpath(out, "afr.bud.csv"))) == 3
        @test length(readlines(joinpath(out, "sln.outer.csv"))) == 1 + sum(s -> s.outer.iterations, sim.results)
        @test length(readlines(joinpath(out, "afr.frac.csv"))) == 3
        # The outer CSV alone builds no inner trace.
        rundir2, _ = _afr_with_csv(joinpath(tmp, "afr2"); ims = false)
        _csv_add_options(joinpath(rundir2, "afr.ims"), "CSV_OUTER_OUTPUT FILEOUT sln.outer.csv")
        sim = initialize(rundir2; backend = TEST_BACKEND, precision = :double, outdir = joinpath(tmp, "out2"))
        @test isnothing(sim.csv.trace) && isnothing(sim.csv.inner)
        ModflowGPU.update(sim)
        @test isnothing(sim.outer.trace)
        finalize_simulation(sim)
    end
end
