# Run parity helpers (T2.8; shared by test/parity/run_test.jl and the synthetic models of
# test/parity/synthetic_test.jl, T3.4): `_run_parity(name, rundir)` runs ModflowGPU on an
# MF6 simulation directory that holds MF6's own output and compares heads, CBC records and
# listing tables per PLAN.md section 4 (criteria in test/parity/run_test.jl's header).
using ModflowGPU: load_simulation, open_head_file, read_head, open_budget_file, read_budget, find_budget_records,
    read_listing_budgets, BudgetList, MF6UnsupportedError, backend_name

const RUN_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# Inner preconditioner of the run parity tests (T4.2): MFGPU_TEST_PRECONDITIONER, default
# `:auto` since T4.5, i.e. the production default (`DEFAULT_PRECONDITIONER`, `:mc_ilu0`;
# `ModflowGPU.run(...; preconditioner)`). `MFGPU_TEST_PRECONDITIONER=jacobi` runs the T4.2
# setting.
const RUN_PRECONDITIONER = Symbol(get(ENV, "MFGPU_TEST_PRECONDITIONER", "auto"))

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
        # List records by package too: a model may have two packages of one type (T6.4).
        pak = r.imeth == 6 ? strip(r.paknam2) : nothing
        j = find_budget_records(a; text = strip(r.text), kstp = r.kstp, kper = r.kper, paknam2 = pak)
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

# Runs both presets on TEST_BACKEND (and on a GPU backend also on the CPU backend) and
# returns per preset (; dh, dq, dl, pd, outer, inner, time, mf6_outer, mf6_inner,
# backend_vs_cpu), plus :single_vs_double. `single_preconditioner` replaces `preconditioner`
# in `:single` (T6.0: ex-gwf-lgrv-gr).
function _run_parity(
        name, rundir; preconditioner::Symbol = RUN_PRECONDITIONER, precond_options::NamedTuple = (;),
        single_preconditioner::Symbol = preconditioner,
    )
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
    stats = Dict{Symbol, Any}()
    for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
        pcname = p.preset === :single ? single_preconditioner : preconditioner
        pc = (; preconditioner = pcname, precond_options)
        mktempdir() do out
            t = @elapsed r = ModflowGPU.run(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = out, pc...)
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
            @info "Run parity: $name, :$(p.preset), $(backend_name(TEST_BACKEND)) ($pcname), $nsteps step(s), $(round(t; digits = 1)) s: max |Δh| = $(round(dh; sigdigits = 3)) m (tol $(round(hdtol; sigdigits = 2))), CBC max |Δq| / total in = $(round(dq; sigdigits = 3)) ($nrec records), listing $(round(dl; sigdigits = 3)) ($ntab tables), max |percent discrepancy| = $(round(pd; sigdigits = 3)); outer/inner iterations $outer/$inner (MF6 $mf6_outer/$mf6_inner)"
            stats[p.preset] = (; dh, dq, dl, pd, outer, inner, time = t, mf6_outer, mf6_inner, backend_vs_cpu = NaN)
            if !(TEST_BACKEND isa CPU)
                rc = ModflowGPU.run(rundir; backend = CPU(), precision = p.preset, write_output = false, pc...)
                d = maximum(abs.(r.heads .- rc.heads))
                p.T === Float64 ? (@test d <= 1.0e-10 * maximum(abs, rc.heads)) : (@test d < 1.0e-4)
                @info "Run $(backend_name(TEST_BACKEND)) vs CPU: $name, :$(p.preset): max |Δh| = $(round(d; sigdigits = 3)) m"
                stats[p.preset] = merge(stats[p.preset], (; backend_vs_cpu = d))
            end
        end
    end
    d = maximum(abs.(heads[:single] .- heads[:double]))
    @test d < 1.0e-3
    @info "Run :single vs :double: $name, max |Δh| = $(round(d; sigdigits = 3)) m (final heads)"
    stats[:single_vs_double] = d
    return stats
end
