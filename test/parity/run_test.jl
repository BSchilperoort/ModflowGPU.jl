# End-to-end parity of `ModflowGPU.run` (T2.8) against MF6 6.8.2 tight-closure runs,
# PLAN.md section 4:
#
# - models: the Tier 0–2 examples that load with `strict = true` and have a tight run in
#   reference/runs/<example>/tight (ex-gwf-hwell-a only with MFGPU_TEST_HWELL=1: 153164
#   cells and 60 steps with the Jacobi preconditioner take minutes), and the synthetic
#   models test/models/fhb_const and test/models/coupling_rde, run with mf6 at test time;
# - heads: every saved step of our HDS file (read back with our reader) vs MF6's:
#   `:double` max |Δh| < max(1e-6 m, 1e-7 × head range), `:single` < 1e-3 m;
# - budgets: every package list record and FLOW-JA-FACE/STO-SS of our CBC vs MF6's
#   (full precision), max |Δq| / total in < 1e-6 (`:double`) or 1e-4 (`:single`); the
#   listing tables (every term's rate in and out, incl. the CHD split, read back with
#   `read_listing_budgets`) vs MF6's, within the same criterion plus the listing rounding;
#   percent discrepancy < 0.01 % in `:single`;
# - outer and inner iterations are reported next to MF6's (mfsim.lst), not tested: the
#   preconditioner is Jacobi until Phase 4 (MF6: ILU0);
# - on a GPU backend, the run is repeated on the CPU backend: `:double` heads within 1e-10
#   relative, `:single` within 1e-4 m.
using ModflowGPU: load_simulation, open_head_file, read_head, open_budget_file, read_budget, find_budget_records,
    read_listing_budgets, BudgetList, MF6UnsupportedError, backend_name

const RUN_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# (kper, kstp) => (outer, inner) from MF6's simulation listing (PRINT_OPTION SUMMARY/ALL).
function _mf6_iterations(mfsim_lst)
    out = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    isfile(mfsim_lst) || return out
    txt = read(mfsim_lst, String)
    for m in eachmatch(r"(\d+) CALLS TO NUMERICAL SOLUTION IN TIME STEP (\d+) STRESS PERIOD (\d+)\s*\n\s*(\d+) TOTAL ITERATIONS", txt)
        out[(parse(Int, m[3]), parse(Int, m[2]))] = (parse(Int, m[1]), parse(Int, m[4]))
    end
    return out
end

_only_file(dir, ext) = (f = filter(x -> endswith(x, ext), readdir(dir; join = true)); length(f) == 1 ? only(f) : "")

# Max |Δh| over all saved steps (the files must hold the same steps) and the head range.
function _head_difference(ours, mf6)
    a, b = open_head_file(ours), open_head_file(mf6)
    @test length(a) == length(b)
    @test [(s.kstp, s.kper) for s in a.steps] == [(s.kstp, s.kper) for s in b.steps]
    @test [s.totim for s in a.steps] ≈ [s.totim for s in b.steps] rtol = 1.0e-12
    d, lo, hi = 0.0, Inf, -Inf
    for k in 1:min(length(a), length(b))
        ha, hb = vec(read_head(a, k)), vec(read_head(b, k))
        act = hb .!= 1.0e30
        @test (ha .== 1.0e30) == .!act
        d = max(d, maximum(abs.(ha[act] .- hb[act])))
        lo, hi = min(lo, minimum(hb[act])), max(hi, maximum(hb[act]))
    end
    return d, hi - lo
end

# Max |Δq| over the MF6 budget records that we write (all but DATA-SPDIS), per step
# relative to that step's total inflow `tin[(kstp, kper)]`.
function _budget_difference(ours, mf6, tin)
    a, b = open_budget_file(ours), open_budget_file(mf6)
    worst = 0.0
    nrec = 0
    for (i, r) in enumerate(b.records)
        strip(r.text) == "DATA-SPDIS" && continue
        j = find_budget_records(a; text = strip(r.text), kstp = r.kstp, kper = r.kper)
        @test length(j) == 1
        length(j) == 1 || continue
        ra = a.records[only(j)]
        @test ra.imeth == r.imeth && (ra.ndim1, ra.ndim2, ra.ndim3) == (r.ndim1, r.ndim2, r.ndim3)
        @test ra.totim ≈ r.totim rtol = 1.0e-12
        @test ra.delt ≈ r.delt rtol = 1.0e-12
        qa, qb = read_budget(a, only(j)), read_budget(b, i)
        if qb isa BudgetList
            @test qa.id1 == qb.id1 && qa.id2 == qb.id2
            @test qa.aux ≈ qb.aux rtol = 1.0e-12
            @test strip(ra.paknam2) == strip(r.paknam2) && ra.auxtxt == r.auxtxt
            va, vb = qa.q, qb.q
        else
            va, vb = vec(qa), vec(qb)
        end
        @test length(va) == length(vb)
        isempty(vb) && continue
        nrec += 1
        worst = max(worst, maximum(abs.(va .- vb)) / tin[(r.kstp, r.kper)])
    end
    return worst, nrec
end

# Listing-table comparison: max over terms of (|Δ| − 2·rounding) / total in.
_listing_round(v) = (v != 0 && (abs(v) < 0.1 || abs(v) >= 9.99999e11)) ? 5.0e-5 * abs(v) : 5.0e-5

function _listing_difference(ours, mf6)
    ta, tb = read_listing_budgets(ours), read_listing_budgets(mf6)
    worst = 0.0
    ntab = 0
    for b in tb
        k = findfirst(x -> (x.kstp, x.kper) == (b.kstp, b.kper), ta)
        isnothing(k) && continue
        a = ta[k]
        ntab += 1
        @test [(t.name, t.package) for t in a.terms] == [(t.name, t.package) for t in b.terms]
        length(a.terms) == length(b.terms) || continue
        for (x, y) in zip(a.terms, b.terms), f in (:rate_in, :rate_out)
            v, w = getfield(x, f), getfield(y, f)
            worst = max(worst, max(abs(v - w) - _listing_round(v) - _listing_round(w), 0.0) / b.total_rate_in)
        end
    end
    return worst, ntab, length(tb)
end

function _run_parity(name, rundir)
    sim = load_simulation(rundir)
    m = only(sim.models)
    mf6_hds = _only_file(rundir, ".hds")
    mf6_cbc = isempty(m.oc.budget_file) ? "" : joinpath(rundir, m.oc.budget_file)
    mf6_lst = joinpath(rundir, isempty(m.list_file) ? splitext(basename(m.file))[1] * ".lst" : m.list_file)
    iters = _mf6_iterations(joinpath(rundir, "mfsim.lst"))
    nsteps = length(ModflowGPU.time_steps(sim.tdis))
    mf6_outer = sum(first, values(iters); init = 0)
    mf6_inner = sum(last, values(iters); init = 0)
    heads = Dict{Symbol, Vector{Float64}}()
    for_each_precision() do p
        mktempdir() do out
            t = @elapsed r = ModflowGPU.run(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = out)
            @test r.converged
            @test r.T === p.T
            heads[p.preset] = r.heads
            dh, range = _head_difference(r.head_file, mf6_hds)
            hdtol = p.T === Float64 ? max(1.0e-6, 1.0e-7 * range) : 1.0e-3
            @test dh < hdtol
            tin = Dict((b.kstp, b.kper) => ModflowGPU.total_rate_in(b) for b in r.budgets)
            btol = p.T === Float64 ? 1.0e-6 : 1.0e-4
            dq, nrec = isempty(mf6_cbc) ? (0.0, 0) : _budget_difference(r.budget_file, mf6_cbc, tin)
            @test dq < btol
            dl, ntab, nmf6 = _listing_difference(r.listing_file, mf6_lst)
            @test dl < btol
            @test ntab == nmf6
            pd = maximum(b -> abs(ModflowGPU.percent_discrepancy(b)), r.budgets)
            p.T === Float32 && @test pd < 0.01
            outer = sum(s -> s.outer.iterations, r.steps)
            inner = sum(s -> s.outer.inner, r.steps)
            @info "Run parity: $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $nsteps step(s), $(round(t; digits = 1)) s: max |Δh| = $(round(dh; sigdigits = 3)) m (tol $(round(hdtol; sigdigits = 2))), CBC max |Δq| / total in = $(round(dq; sigdigits = 3)) ($nrec records), listing $(round(dl; sigdigits = 3)) ($ntab tables), max |percent discrepancy| = $(round(pd; sigdigits = 3)); outer/inner iterations $outer/$inner (MF6 $mf6_outer/$mf6_inner)"
            if !(TEST_BACKEND isa CPU)
                rc = ModflowGPU.run(rundir; backend = CPU(), precision = p.preset, write_output = false)
                d = maximum(abs.(r.heads .- rc.heads))
                p.T === Float64 ? (@test d <= 1.0e-10 * maximum(abs, rc.heads)) : (@test d < 1.0e-4)
                @info "Run $(backend_name(TEST_BACKEND)) vs CPU: $name, :$(p.preset): max |Δh| = $(round(d; sigdigits = 3)) m"
            end
        end
    end
    d = maximum(abs.(heads[:single] .- heads[:double]))
    @test d < 1.0e-3
    @info "Run :single vs :double: $name, max |Δh| = $(round(d; sigdigits = 3)) m (final heads)"
    return nothing
end

function _run_models(f)
    hwell = get(ENV, "MFGPU_TEST_HWELL", "") == "1"
    for (name, _) in in_scope_examples(; tiers = 0:2)
        rundir = joinpath(REFERENCE_DIR, "runs", name, "tight")
        if !isdir(rundir)
            @info "Skipping $name in the run parity test: $rundir not found"
            continue
        end
        if name == "ex-gwf-hwell-a" && !hwell
            @info "Skipping $name in the run parity test (153164 cells, 60 steps; set MFGPU_TEST_HWELL=1)"
            continue
        end
        try
            load_simulation(rundir; strict = true)
        catch err
            err isa MF6UnsupportedError || rethrow()
            @info "Skipping $name in the run parity test (not loadable): $(sprint(showerror, err))"
            continue
        end
        f(name, rundir)
    end
    if !isfile(RUN_MF6)
        @info "Skipping the synthetic models in the run parity test: mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
        return nothing
    end
    for name in ("fhb_const", "coupling_rde")
        mktempdir() do tmp
            rundir = joinpath(tmp, name)
            cp(joinpath(@__DIR__, "..", "models", name), rundir)
            out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
            @test occursin("Normal termination", out)
            f(name, rundir)
        end
    end
    return nothing
end

@testset "Run parity vs MF6 (heads, budgets, listing)" begin
    nmodels = Ref(0)
    _run_models() do name, rundir
        nmodels[] += 1
        @testset "$name" begin
            _run_parity(name, rundir)
        end
    end
    @info "Run parity test: $(nmodels[]) model(s)"
end
