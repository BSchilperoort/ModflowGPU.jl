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

# Max |Δh| over all saved steps (the files must hold the same steps) and the head range;
# dry cells (HDRY, T6.1) must be the same and are left out of both. With `bot` (the cell
# bottoms by user node, Newton models, T6.2) cells whose MF6 head is below the bottom are
# dry in the Newton sense: no saturation, no outflow, and a head that any value between the
# highest neighbouring wet head and the bottom satisfies (MF6 ends where its iterations
# stop, e.g. ex-gwf-bump-p01c 1.25e-8 m below the bottom, ModflowGPU 1.25e-6 m). Ours must
# be below the bottom too (within `drytol`); they are left out of the difference and the
# range. Returns also the number of such cell-steps.
#
# `shift = true` (T6.5, `_null_space_model`): heads are defined up to a constant, so the
# mean difference of each step is removed first (and returned as the largest such shift).
function _head_difference(ours, mf6; bot = nothing, drytol = 0.0, shift::Bool = false)
    a, b = open_head_file(ours), open_head_file(mf6)
    @test length(a) == length(b)
    @test [(s.kstp, s.kper) for s in a.steps] == [(s.kstp, s.kper) for s in b.steps]
    @test [s.totim for s in a.steps] ≈ [s.totim for s in b.steps] rtol = 1.0e-12
    d, lo, hi = 0.0, Inf, -Inf
    ndry = 0
    cmax = 0.0
    for k in 1:min(length(a), length(b))
        ha, hb = vec(read_head(a, k)), vec(read_head(b, k))
        act = (hb .!= 1.0e30) .& (hb .!= -1.0e30)
        @test (ha .== 1.0e30) == (hb .== 1.0e30)
        @test (ha .== -1.0e30) == (hb .== -1.0e30)
        if !isnothing(bot)
            dry = act .& (hb .< bot)
            ndry += count(dry)
            @test all(ha[dry] .< bot[dry] .+ drytol)
            act .&= .!dry
        end
        c = shift && any(act) ? sum(ha[act] .- hb[act]) / count(act) : 0.0
        cmax = max(cmax, abs(c))
        d = max(d, maximum(abs.(ha[act] .- hb[act] .- c); init = 0.0))
        any(act) && ((lo, hi) = (min(lo, minimum(hb[act])), max(hi, maximum(hb[act]))))
    end
    return d, hi - lo, ndry, cmax
end

_spec_kind(::ModflowGPU.BoundarySpec{K}) where {K} = K


# Max |a − b| (`shift`: after removing the mean difference).
_maxdiff(a, b, shift) = (c = shift ? sum(a .- b) / length(a) : 0.0; maximum(abs.(a .- b .- c); init = 0.0))

"""
    _null_space_model(m::GwfSpec, nper) -> Bool

Whether the heads of model `m` are defined only up to a constant (T6.5, ex-gwf-whirl): every
period steady state and no head-dependent boundary (WEL and RCH only), so the matrix is
singular and its null space is the constant heads. MF6's heads and ours differ by the
constant the linear solves leave in that direction (ex-gwf-whirl: 8.0e-6 m, with a spread
of 1e-10 m); the parity test compares them after removing the mean difference.
"""
function _null_space_model(m, nper)
    all(b -> _spec_kind(b) in (:WEL, :RCH), m.boundaries) || return false
    isnothing(m.sto) && return true
    return !any(k -> ModflowGPU.is_transient(m.sto, k), 1:nper)
end

# --- `:single` budget bound from Float32 rounding of the cell flows (T6.1, docs/dev/precision.md)
#
# A flow of the engine is `q = C·(h'_m − h'_n + Δr)` (storage: `κ·(h'old − h')`, `κ = rho1 +
# rho2`) from offset heads rounded to Float32, so it differs from MF6's by up to
# `C·(eps32·(|h'_n| + |h'_m| + |Δr|) + |δh_n| + |δh_m|)`, where `δh` is the difference
# between the converged heads (the Float32 closures, checked separately against 1e-3 m).
# With large conductances or storage coefficients (rad-disu: C up to 5.5e7 ft²/d and
# Sy·area/delt in the first steps; maw-p03c: total inflow 18 m³/d) this exceeds 1e-4 ×
# total in although the heads agree to 1e-5 m. `_rounding_bounds` gives these bounds per
# FLOW-JA-FACE entry and STO-SS/STO-SY cell (C from MF6's flow and head difference, or
# condsat when MF6's heads are equal; 4 × eps32 for the rounding), and per step their sum
# `total` (the bound of a listing term and of the in − out imbalance).
function _rounding_bounds(rundir, ours_hds, mf6_hds)
    sim = load_simulation(rundir)
    m = only(sim.models)
    conn = ModflowGPU.build_connectivity(m)
    s = ModflowGPU.npf_setup(m, conn)
    a, b = open_head_file(ours_hds), open_head_file(mf6_hds)
    ε = 4 * eps(Float32)
    heads = Dict{Tuple{Int, Int}, Tuple{Vector{Float64}, Vector{Float64}}}()
    keys_ = Tuple{Int, Int}[]
    for k in 1:min(length(a), length(b))
        st = b.steps[k]
        key = (Int(st.kstp), Int(st.kper))
        heads[key] = (vec(read_head(a, k))[conn.nodeuser], vec(read_head(b, k))[conn.nodeuser])
        push!(keys_, key)
    end
    sto = m.sto
    # storage bound by user node for step `key`, time step length `delt`
    function storage_bound(key, delt, nu)
        (isnothing(sto) || !haskey(heads, key)) && return nothing
        ho, hm = heads[key]
        i = findfirst(==(key), keys_)
        hoold, hmold = (isnothing(i) || i == 1) ? (nothing, nothing) : heads[keys_[i - 1]]
        bnd = zeros(nu)
        for n in 1:conn.nodes
            hm[n] == -1.0e30 && continue
            u = conn.nodeuser[n]
            thk = sto.storagecoefficient ? 1.0 : conn.top[n] - conn.bot[n]
            κ = (sto.ss[u] * thk + (sto.iconvert[u] != 0 && !isnothing(sto.sy) ? sto.sy[u] : 0.0)) * conn.area[n] / delt
            old = !isnothing(hmold) && hmold[n] != -1.0e30
            hold = old ? hmold[n] : hm[n]
            dold = old ? abs(hoold[n] - hmold[n]) : 0.0
            bnd[u] = κ * (ε * (abs(hm[n] - s.r[n]) + abs(hold - s.r[n])) + abs(ho[n] - hm[n]) + dold)
        end
        return bnd
    end
    # flowja bound of step `key` with MF6's flows `q` (nja; `nothing` when MF6 saved no
    # budget, T6.2: C = condsat, an upper bound without VARIABLECV); the diagonal (the
    # cell's net flow) also gets the cell's storage bound (`delt`: the step length)
    function flowja_bound(key, q, delt)
        haskey(heads, key) || return nothing
        ho, hm = heads[key]
        bnd = zeros(conn.nja)
        for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            mm = conn.ja[ii]
            (hm[n] == -1.0e30 || hm[mm] == -1.0e30) && continue
            dh = hm[mm] - hm[n]
            C = (!isnothing(q) && dh != 0) ? abs(q[ii] / dh) : s.condsat[conn.jas[ii]]
            dr = s.r[mm] - s.r[n]
            bnd[ii] = C * (ε * (abs(hm[n] - s.r[n]) + abs(hm[mm] - s.r[mm]) + abs(dr)) + abs(ho[n] - hm[n]) + abs(ho[mm] - hm[mm]))
        end
        sb = storage_bound(key, delt, Int(conn.nodesuser))
        for n in 1:conn.nodes  # diagonal: the cell's net flow
            bnd[conn.ia[n]] = sum(bnd[ii] for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1); init = 0.0) +
                (isnothing(sb) ? 0.0 : sb[conn.nodeuser[n]])
        end
        return bnd
    end
    return (; flowja_bound, storage_bound, idiag = conn.ia[1:conn.nodes], nodesuser = Int(conn.nodesuser))
end

# Max |Δq| over the MF6 budget records that we write (all but DATA-SPDIS), per step
# relative to that step's total inflow `tin[(kstp, kper)]`. With `bounds`
# (`_rounding_bounds`, `:single`, T6.1) the FLOW-JA-FACE, STO-SS and STO-SY values are
# compared after subtracting their bounds, and the per-step sums of those bounds are
# returned as well (the allowance of the listing terms and the percent discrepancy).
function _budget_difference(ours, mf6, tin; bounds = nothing)
    a, b = open_budget_file(ours), open_budget_file(mf6)
    worst = 0.0
    nrec = 0
    btot = Dict{Tuple{Int, Int}, Float64}()
    for (i, r) in enumerate(b.records)
        strip(r.text) == "DATA-SPDIS" && continue
        # List records by package too: a model may have two packages of one type (T6.4).
        pak = r.imeth == 6 ? strip(r.paknam2) : nothing
        j = find_budget_records(a; text = strip(r.text), kstp = r.kstp, kper = r.kper, paknam2 = pak)
        # T6.6: a periodic exchange writes two lists with the same text and package to its
        # model's file (from either side): matched in file order.
        jb = find_budget_records(b; text = strip(r.text), kstp = r.kstp, kper = r.kper, paknam2 = pak)
        @test length(j) == length(jb)
        length(j) == length(jb) || continue
        j = [j[findfirst(==(i), jb)]]
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
        d = abs.(va .- vb)
        if !isnothing(bounds)
            key = (Int(r.kstp), Int(r.kper))
            t = strip(r.text)
            tolv = t == "FLOW-JA-FACE" ? bounds.flowja_bound(key, vb, r.delt) :
                t in ("STO-SS", "STO-SY") ? bounds.storage_bound(key, r.delt, length(vb)) : nothing
            if !isnothing(tolv)
                d = max.(d .- tolv, 0.0)
                # FLOW-JA-FACE: each connection once (the diagonals hold the cell sums,
                # with storage, which the STO records add).
                if t == "FLOW-JA-FACE"
                    off = sum(tolv) - sum(tolv[ii] for ii in bounds.idiag)
                    btot[key] = get(btot, key, 0.0) + off / 2
                else
                    btot[key] = get(btot, key, 0.0) + sum(tolv)
                end
            end
        end
        worst = max(worst, maximum(d) / tin[(r.kstp, r.kper)])
    end
    return worst, nrec, btot
end

# Listing-table comparison: max over terms of (|Δ| − 2·rounding) / total in.
_listing_round(v) = (v != 0 && (abs(v) < 0.1 || abs(v) >= 9.99999e11)) ? 5.0e-5 * abs(v) : 5.0e-5

function _listing_difference(ours, mf6; extra = Dict{Tuple{Int, Int}, Float64}())
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
            ex = get(extra, (Int(b.kstp), Int(b.kper)), 0.0)
            worst = max(worst, max(abs(v - w) - _listing_round(v) - _listing_round(w) - ex, 0.0) / b.total_rate_in)
        end
    end
    return worst, ntab, length(tb)
end

# (time step, outer iteration) pairs in which backtracking reduced the head change: from
# MF6's outer table rows "Backtracking <kiter> <flag> <count> ..." (count > 0; a new time
# step starts at kiter 1), or from the `BacktrackRecord`s of a ModflowGPU run.
function _backtracked_iterations(lst::AbstractString)
    out = Tuple{Int, Int}[]
    istep = 0
    for x in eachmatch(r"\n\s*Backtracking\s+(\d+)\s+(\d+)\s+(\d+)", lst)
        k = parse(Int, x[1])
        k == 1 && (istep += 1)
        parse(Int, x[3]) > 0 && push!(out, (istep, k))
    end
    return out
end

_backtracked_iterations(steps::AbstractVector) =
    [(i, b.kiter) for (i, s) in enumerate(steps) for b in s.outer.backtracking if b.count > 0]

# --- Observations (T6.7)

# The observation output files of model `m`: (file as written, observation types by column).
function _obs_outputs(m)
    out = Tuple{String, Vector{String}}[]
    specs = Any[m.obs]
    append!(specs, [b.obs for b in m.boundaries])
    for os in specs
        isnothing(os) && continue
        for o in os.outputs
            push!(out, (o.file, [e.type for e in o.observations]))
        end
    end
    return out
end

"""
    _obs_difference(m, dir_a, dir_b, tin; htol, ftol) -> (dh, dq, nvalues)

Compare the observation files of model `m` in `dir_a` (ours) with those in `dir_b` (MF6's,
or the CPU run's): the same names and times (rtol 1e-12), HEAD and DRAWDOWN within `htol`
(length units), every other type (flows) within `ftol` × the larger of the column's
largest |value| and the step's total inflow `tin[i]` (row `i`; the scale of the budget
criterion: ex-gwf-maw-p03c observes vertical flows of 1e-4 m³/d next to a total inflow of
18 m³/d), and `OBSNODATA` in the same places. Returns the largest head difference, the
largest scaled flow difference and the number of values compared.
"""
function _obs_difference(m, dir_a, dir_b, tin; htol, ftol)
    dh, dq, nv = 0.0, 0.0, 0
    for (file, types) in _obs_outputs(m)
        a = ModflowGPU.read_observations(joinpath(dir_a, file))
        b = ModflowGPU.read_observations(joinpath(dir_b, file))
        @test a.names == b.names
        @test a.times ≈ b.times rtol = 1.0e-12
        size(a.values) == size(b.values) || (@test size(a.values) == size(b.values); continue)
        for (j, t) in enumerate(types)
            va, vb = a.values[:, j], b.values[:, j]
            nod = vb .== ModflowGPU.OBSNODATA
            @test (va .== ModflowGPU.OBSNODATA) == nod
            keep = .!nod
            isempty(vb[keep]) && continue
            nv += count(keep)
            if t in ("HEAD", "DRAWDOWN")
                dh = max(dh, maximum(abs.(va[keep] .- vb[keep])))
            else
                cmax = maximum(abs, vb[keep])
                for i in findall(keep)
                    sc = max(cmax, tin[i])
                    dq = max(dq, sc == 0 ? abs(va[i] - vb[i]) : abs(va[i] - vb[i]) / sc)
                end
            end
        end
    end
    @test dh < htol
    @test dq < ftol
    return dh, dq, nv
end

# Runs both presets on TEST_BACKEND (and on a GPU backend also on the CPU backend) and
# returns per preset (; dh, dq, dl, pd, outer, inner, time, mf6_outer, mf6_inner,
# backend_vs_cpu), plus :single_vs_double. `single_preconditioner` replaces `preconditioner`
# in `:single` (T6.0: ex-gwf-lgrv-gr). `single_budget_tol` replaces the `:single` budget
# criterion 1e-4 × total in (T6.4: rad_disu_confined, docs/dev/precision.md).
# `backtracking = true` (T6.2a) also checks that the run backtracks in the same (time
# step, outer iteration)s as MF6. `presets` (T6.2): the presets to run (ex-gwf-nwt-p03b
# leaves out `:single`, see `run_parity_options`). `single_obs_tol` (T6.7): the `:single`
# criterion of the observed flows (× the step's total inflow, `_obs_difference`). Newton models (T6.2): cells whose MF6
# head is below the bottom are compared as dry (`_head_difference`), and also left out of
# the GPU vs CPU and `:single` vs `:double` head comparisons (their heads depend on the
# iteration path).
# T6.9: the cell-conversion blocks of a listing file (`sgwf_npf_wdmsg`): header => lines.
function _conversion_blocks(path)
    lines = readlines(path)
    out = Pair{String, Vector{String}}[]
    i = 1
    while i <= length(lines)
        if occursin("CELL CONVERSIONS FOR", lines[i])
            hdr = strip(lines[i])
            body = String[]
            i += 1
            while i <= length(lines) && occursin(r"^    \s*(WET|DRY)\s", lines[i])
                push!(body, lines[i])
                i += 1
            end
            push!(out, hdr => body)
        else
            i += 1
        end
    end
    return out
end

function _run_parity(
        name, rundir; preconditioner::Symbol = RUN_PRECONDITIONER, precond_options::NamedTuple = (;),
        single_preconditioner::Symbol = preconditioner, single_budget_tol::Float64 = 1.0e-4,
        double_budget_tol::Float64 = 1.0e-6, single_rounding_bound::Bool = false,
        backtracking::Bool = false, presets = SOLVER_TEST_PRESETS, single_obs_tol::Float64 = 1.0e-4,
    )
    sim = load_simulation(rundir)
    # T6.6: several models (one IMS solution with GWF-GWF exchanges): every model's head,
    # budget and listing files against MF6's, and the largest differences reported.
    models = sim.models
    multi = length(models) > 1
    m = first(models)
    # Newton models: the cell bottoms by user node (dry cells) and by reduced node.
    conn = m.newton ? ModflowGPU.build_connectivity(m) : nothing
    botu = m.newton ? (b = fill(-Inf, Int(conn.nodesuser)); b[conn.nodeuser] .= conn.bot; b) : nothing
    wet(h) = isnothing(conn) ? trues(length(h)) : h .>= conn.bot
    shift = all(x -> _null_space_model(x, ModflowGPU.nper(sim.tdis)), models)
    mf6_files(x) = (
        hds = multi ? joinpath(rundir, x.oc.head_file) : _only_file(rundir, ".hds"),
        cbc = isempty(x.oc.budget_file) ? "" : joinpath(rundir, x.oc.budget_file),
        lst = joinpath(rundir, isempty(x.list_file) ? splitext(basename(x.file))[1] * ".lst" : x.list_file),
    )
    iters = _mf6_iterations(joinpath(rundir, "mfsim.lst"))
    nsteps = length(ModflowGPU.time_steps(sim.tdis))
    mf6_outer = sum(first, values(iters); init = 0)
    mf6_inner = sum(last, values(iters); init = 0)
    mf6_bt = backtracking ? _backtracked_iterations(read(joinpath(rundir, "mfsim.lst"), String)) : nothing
    heads = Dict{Symbol, Vector{Float64}}()
    stats = Dict{Symbol, Any}()
    hasobs = any(x -> !isempty(_obs_outputs(x)), models)
    for_each_precision(; presets) do p
        pcname = p.preset === :single ? single_preconditioner : preconditioner
        pc = (; preconditioner = pcname, precond_options)
        mktempdir() do out
            t = @elapsed r = ModflowGPU.run(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = out, pc...)
            @test r.converged
            @test r.T === p.T
            heads[p.preset] = r.heads
            @test length(r.models) == length(models)
            # T6.6: then the observation files of the GWF-GWF exchanges.
            exobs = [o.file for ex in sim.exchanges if !isnothing(ex.obs) for o in ex.obs.outputs]
            @test length(r.obs_files) == sum(x -> length(_obs_outputs(x)), models) + length(exobs)
            for file in exobs
                a, b = ModflowGPU.read_observations(joinpath(out, file)), ModflowGPU.read_observations(joinpath(rundir, file))
                @test a.names == b.names
                @test a.times ≈ b.times rtol = 1.0e-12
                sc = max(maximum(abs, b.values; init = 0.0), floatmin())
                @test maximum(abs.(a.values .- b.values); init = 0.0) / sc < (p.T === Float64 ? 1.0e-6 : single_obs_tol)
            end
            dh = dq = dl = pd = pdx = hshift = 0.0
            ndry = nrec = ntab = 0
            obs = nothing
            hdtols = Float64[]
            tinss = Dict{String, Vector{Float64}}()
            for x in models
                mo = r.models[findfirst(y -> y.name == x.name, r.models)]
                f = mf6_files(x)
                budgets = multi ? filter(b -> b.model == x.name, r.budgets) : r.budgets
                @test length(budgets) == nsteps
                dhk, range, ndryk, hshiftk = _head_difference(mo.head_file, f.hds; bot = botu, drytol = p.T === Float64 ? 1.0e-6 : 1.0e-3, shift)
                hdtol = p.T === Float64 ? max(1.0e-6, 1.0e-7 * range) : 1.0e-3
                push!(hdtols, hdtol)
                @test dhk < hdtol
                dh, ndry, hshift = max(dh, dhk), ndry + ndryk, max(hshift, hshiftk)
                # T6.7: observation files vs MF6's (heads within the head-file criterion, flows
                # relative to the column's largest value: 1e-6 in :double, 1e-4 in :single).
                tins = [ModflowGPU.total_rate_in(b) for b in budgets]
                tinss[x.name] = tins
                if !isempty(_obs_outputs(x))
                    o = _obs_difference(x, out, rundir, tins; htol = hdtol, ftol = p.T === Float64 ? 1.0e-6 : single_obs_tol)
                    obs = isnothing(obs) ? o : (max(obs[1], o[1]), max(obs[2], o[2]), obs[3] + o[3])
                end
                tin = Dict((b.kstp, b.kper) => ModflowGPU.total_rate_in(b) for b in budgets)
                btol = p.T === Float64 ? double_budget_tol : single_budget_tol
                # `:single` with `single_rounding_bound` (T6.1, named models only): budget
                # differences beyond the Float32 rounding of the flows and the head difference
                # (`_rounding_bounds`); otherwise the raw differences.
                rb = p.T === Float32 && single_rounding_bound ? _rounding_bounds(rundir, mo.head_file, f.hds) : nothing
                dqk, nreck, btot = isempty(f.cbc) ? (0.0, 0, Dict{Tuple{Int, Int}, Float64}()) : _budget_difference(mo.budget_file, f.cbc, tin; bounds = rb)
                if !isnothing(rb) && isempty(f.cbc)
                    # No MF6 budget file (T6.2, ex-gwf-zaidel): the bound of each step from the
                    # connections (C = condsat) and storage, for the listing and the imbalance.
                    for st in ModflowGPU.time_steps(sim.tdis)
                        key = (st.kstp, st.kper)
                        tolv = rb.flowja_bound(key, nothing, st.delt)
                        isnothing(tolv) && continue
                        sb = rb.storage_bound(key, st.delt, rb.nodesuser)
                        btot[key] = (sum(tolv) - sum(tolv[ii] for ii in rb.idiag)) / 2 + (isnothing(sb) ? 0.0 : sum(sb))
                    end
                end
                @test dqk < btol
                dlk, ntabk, nmf6 = _listing_difference(mo.listing_file, f.lst; extra = btot)
                @test dlk < btol
                @test ntabk == nmf6
                # T6.9: the cell-conversion messages as MF6 writes them (`:double`: the same
                # iterations convert the same cells; `:single` reported).
                cva, cvb = _conversion_blocks(mo.listing_file), _conversion_blocks(f.lst)
                p.T === Float64 && @test cva == cvb
                (isempty(cva) && isempty(cvb)) || @info "Cell conversions: $name/$(x.name), :$(p.preset): $(length(cva)) blocks (MF6 $(length(cvb))), identical: $(cva == cvb)"
                # Percent discrepancy beyond the bound of the in − out imbalance (`:single`).
                pdxk = maximum(b -> max(abs(ModflowGPU.percent_discrepancy(b)) - 100 * get(btot, (b.kstp, b.kper), 0.0) / ModflowGPU.total_rate_in(b), 0.0), budgets)
                pdk = maximum(b -> abs(ModflowGPU.percent_discrepancy(b)), budgets)
                p.T === Float32 && @test pdxk < 0.01
                dq, dl, pd, pdx = max(dq, dqk), max(dl, dlk), max(pd, pdk), max(pdx, pdxk)
                nrec += nreck
                ntab += ntabk
            end
            hdtol = maximum(hdtols)
            outer = sum(s -> s.outer.iterations, r.steps)
            inner = sum(s -> s.outer.inner, r.steps)
            if backtracking
                # T6.2a: backtracking in the same (step, outer iteration)s as MF6.
                bt = _backtracked_iterations(r.steps)
                @test bt == mf6_bt
                @info "Backtracking: $name, :$(p.preset), $(backend_name(TEST_BACKEND)): (step, outer iteration) $bt (MF6 $mf6_bt)"
            end
            mtxt = multi ? " ($(length(models)) models)" : ""
            @info "Run parity: $name$mtxt, :$(p.preset), $(backend_name(TEST_BACKEND)) ($pcname), $nsteps step(s), $(round(t; digits = 1)) s: max |Δh| = $(round(dh; sigdigits = 3)) m$(shift ? " after removing a constant shift of $(round(hshift; sigdigits = 3)) m (null space)" : "") (tol $(round(hdtol; sigdigits = 2))$(ndry > 0 ? "; $ndry Newton-dry cell-steps" : "")), CBC max |Δq| / total in = $(round(dq; sigdigits = 3)) ($nrec records), listing $(round(dl; sigdigits = 3)) ($ntab tables), max |percent discrepancy| = $(round(pd; sigdigits = 3))$(p.T === Float32 ? " (beyond the rounding bound $(round(pdx; sigdigits = 3)))" : ""); outer/inner iterations $outer/$inner (MF6 $mf6_outer/$mf6_inner)"
            isnothing(obs) || @info "Run parity, observations: $name, :$(p.preset), $(backend_name(TEST_BACKEND)): max |Δ| heads $(round(obs[1]; sigdigits = 3)) m, flows $(round(obs[2]; sigdigits = 3)) × scale ($(obs[3]) values in $(length(r.obs_files)) file(s))"
            stats[p.preset] = (; dh, dq, dl, pd, outer, inner, time = t, mf6_outer, mf6_inner, backend_vs_cpu = NaN, obs)
            if !(TEST_BACKEND isa CPU)
                mktempdir() do cout
                    # The CPU run writes output only when there are observations to compare.
                    rc = ModflowGPU.run(rundir; backend = CPU(), precision = p.preset, write_output = hasobs, outdir = cout, pc...)
                    w = wet(rc.heads)
                    d = _maxdiff(r.heads[w], rc.heads[w], shift)
                    p.T === Float64 ? (@test d <= 1.0e-10 * maximum(abs, rc.heads)) : (@test d < 1.0e-4)
                    @info "Run $(backend_name(TEST_BACKEND)) vs CPU: $name, :$(p.preset): max |Δh| = $(round(d; sigdigits = 3)) m"
                    stats[p.preset] = merge(stats[p.preset], (; backend_vs_cpu = d))
                    if hasobs
                        # T6.7: :double 1e-10 relative, :single heads 1e-4 m, flows 1e-4 × scale.
                        hr = maximum(abs, rc.heads)
                        for x in models
                            isempty(_obs_outputs(x)) && continue
                            oc = _obs_difference(x, out, cout, tinss[x.name]; htol = p.T === Float64 ? 1.0e-10 * hr : 1.0e-4, ftol = p.T === Float64 ? 1.0e-10 : 1.0e-4)
                            @info "Run $(backend_name(TEST_BACKEND)) vs CPU, observations: $name, :$(p.preset): max |Δ| heads $(round(oc[1]; sigdigits = 3)) m, flows $(round(oc[2]; sigdigits = 3)) × scale"
                        end
                    end
                end
            end
        end
    end
    if haskey(heads, :single) && haskey(heads, :double)
        w = wet(heads[:double])
        d = _maxdiff(heads[:single][w], heads[:double][w], shift)
        @test d < 1.0e-3
        @info "Run :single vs :double: $name, max |Δh| = $(round(d; sigdigits = 3)) m (final heads)"
        stats[:single_vs_double] = d
    end
    return stats
end

# --- The models of the run parity test (test/parity/run_test.jl), also run by
# docs/parity/collect.jl and checked against docs/src/assets/parity.toml (group api).

"Sizes of benchmark/synthetic.jl in the synthetic parity test (test/parity/synthetic_test.jl)."
const SYNTHETIC_PARITY_MODELS = ("tiny", "indus")

"Synthetic models in test/models/ that the run parity test runs with mf6 at test time."
const RUN_PARITY_TEST_MODELS = ("ts_tas", "coupling_rde", "wetdry", "disu_staggered", "afr", "newton")

"""
    run_parity_models() -> Vector{Pair{String, Any}}

The models of the run parity test, in run order: `name => source`, where `source` is the
reference run directory of an MF6 example (a `String`; `parity_examples()` with a
converged reference run that strict loading accepts), `(:test_model, name)` for
`RUN_PARITY_TEST_MODELS`, `:rad_disu_confined` or `:capture_saved` (T6.2; when their
reference runs exist). The last three need mf6 (`RUN_MF6`); `with_parity_rundir` prepares
them. Examples left out are logged with the reason.
"""
function run_parity_models()
    hwell = get(ENV, "MFGPU_TEST_HWELL", "1") != "0"
    out = Pair{String, Any}[]
    for (name, _) in parity_examples()
        rundir = reference_rundir(name)
        if isnothing(rundir)
            @info "Skipping $name in the run parity test: no converged reference run in $(joinpath(REFERENCE_DIR, "runs", name))"
            continue
        end
        if name == "ex-gwf-hwell-a" && !hwell
            @info "Skipping $name in the run parity test (MFGPU_TEST_HWELL=0)"
            continue
        end
        try
            load_simulation(rundir; strict = true)
        catch err
            err isa MF6UnsupportedError || rethrow()
            @info "Skipping $name in the run parity test (not loadable): $(sprint(showerror, err))"
            continue
        end
        push!(out, name => rundir)
    end
    for name in filter(test_model_selected, RUN_PARITY_TEST_MODELS)
        push!(out, name => (:test_model, name))
    end
    # DISU against MF6 (T6.4): ex-gwf-rad-disu with confined storage.
    if !test_model_selected("rad_disu_confined")
    elseif isdir(joinpath(REFERENCE_DIR, "runs", "ex-gwf-rad-disu", "tight-relative"))
        push!(out, "rad_disu_confined" => :rad_disu_confined)
    else
        @info "Skipping rad_disu_confined in the run parity test: reference/runs/ex-gwf-rad-disu not found"
    end
    # ex-gwf-capture (T6.2) saves no heads: run with an OC that saves them.
    if !test_model_selected("capture_saved")
    elseif isdir(joinpath(REFERENCE_DIR, "runs", "ex-gwf-capture", "tight"))
        push!(out, "capture_saved" => :capture_saved)
    else
        @info "Skipping capture_saved in the run parity test: reference/runs/ex-gwf-capture not found"
    end
    return out
end

"""
    with_parity_rundir(f, name, source)

Call `f(rundir)` with the simulation directory of a `run_parity_models()` entry: the
reference run itself, or a temporary copy run with mf6 first. Returns `nothing` without
calling `f` (logged) when such a model needs mf6 and `RUN_MF6` does not exist.
"""
function with_parity_rundir(f, name, source)
    source isa AbstractString && return f(source)
    if !isfile(RUN_MF6)
        @info "Skipping $name in the run parity test: mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
        return nothing
    end
    return mktempdir() do tmp
        rundir = joinpath(tmp, name)
        if source === :rad_disu_confined
            rad_disu_confined(rundir)
        elseif source === :capture_saved
            capture_saved(rundir)
        else
            cp(joinpath(@__DIR__, "models", last(source)), rundir)
        end
        out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
        @test occursin("Normal termination", out)
        f(rundir)
    end
end

"Run `f(name, rundir)` for every model of the run parity test (`run_parity_models`)."
function _run_models(f)
    for (name, source) in run_parity_models()
        with_parity_rundir(rundir -> f(name, rundir), name, source)
    end
    return nothing
end

"""
    run_parity_options(name) -> NamedTuple

The keyword arguments of `_run_parity` for model `name` of the run parity test: the
per-example preconditioners and criteria (the reasons are in the comments and in
docs/dev/precision.md).
"""
function run_parity_options(name)
    # ex-gwf-hwell-a: Jacobi BiCGSTAB does not converge in :single (T2.8); AMG does
    # (T4.3), unless MFGPU_TEST_PRECONDITIONER chooses another preconditioner.
    default_pc = !haskey(ENV, "MFGPU_TEST_PRECONDITIONER")
    # rad_disu_confined (T6.4): radial DISU grid without fixed heads, nearly singular
    # in the late (long) steps; MC-ILU0 BiCGSTAB stalls near 1e-8 m from step 21 on,
    # AMG converges in 2 outer iterations per step, as MF6 with ILUT does.
    hwell_amg = name in ("ex-gwf-hwell-a", "rad_disu_confined", "ex-gwf-rad-disu") && default_pc
    pc = hwell_amg ? :amg : RUN_PRECONDITIONER
    # ex-gwf-lgrv-gr in :single: Float32 CG with MC-ILU0 stops 6.1e-3 m from the exact
    # heads (the correction solves converge too slowly on this badly conditioned steady
    # state); with AMG 6.7e-6 m (T6.0, docs/dev/precision.md).
    spc = name == "ex-gwf-lgrv-gr" && default_pc ? :amg : pc
    # :single budgets (T6.1): for the models below only, differences beyond the
    # Float32 rounding of the cell flows and the head difference (`_rounding_bounds`)
    # must be below 1e-4 × total in (replaces the 5e-4 of rad_disu_confined, T6.4); every
    # other model keeps the raw criteria. Raw `:single` values there: rad-disu CBC 0.39 ×
    # total in and pd 48.5 %, maw-p03c 1.0e-3 and 0.60 %, advtidal pd 1.9e-2 %
    # (docs/dev/precision.md).
    # T6.2: ex-gwf-zaidel (K = 1e-4 m/d, head differences of 6 cm per cell next to heads of
    # 13 m: pd 1.3e-2 % raw, no MF6 budget file, so the bound takes C = condsat).
    srb = name in ("rad_disu_confined", "ex-gwf-rad-disu", "ex-gwf-maw-p03c", "ex-gwf-advtidal", "ex-gwf-zaidel")
    # ex-gwf-rad-disu in :double/:mixed: STO-SY of cell 21 in step 2 (Sy·area/delt is
    # large: Δt = 9.4e-5 d; heads at the cell top) turns the 2.5e-10 ft head difference
    # to MF6's tight-relative reference into 6.0e-6 × total in; criterion 1e-5 there
    # (docs/dev/precision.md).
    dbt = name == "ex-gwf-rad-disu" ? 1.0e-5 : 1.0e-6
    # T6.2: test/models/newton (COMPLEXITY COMPLEX with NEWTON) backtracks in the same outer
    # iterations as MF6.
    bt = name == "newton"
    # ex-gwf-nwt-p03b in :single (T6.2): the Newton iterations do not converge in Float32
    # (500 outer iterations, with MC-ILU0 and with AMG). MF6's own path is erratic (inner
    # solves at INNER_MAXIMUM in outer iterations 1–3, changes up to 1420 m, 159 outer
    # iterations in :double); in Float32 the unconverged inner solves end far off and the
    # iterations wander between states with changes of 10–1000 m (docs/dev/precision.md).
    # :mixed and :double converge.
    presets = name == "ex-gwf-nwt-p03b" ? (:mixed, :double) : SOLVER_TEST_PRESETS
    name == "ex-gwf-nwt-p03b" && @info "Run parity: $name without :single (the Newton iterations do not converge in Float32; docs/dev/precision.md)"
    # T6.7: ex-gwf-maw-p03c observes the flows next to its (former) well, where heads are
    # flat (10.0001 m) and conductances large: in :single the Float32 rounding of these flows
    # is 2.2e-4 × the total inflow (the raw :single CBC difference there is 1.0e-3,
    # docs/dev/precision.md), with heads within 1.6e-6 m.
    sot = name == "ex-gwf-maw-p03c" ? 1.0e-3 : 1.0e-4
    return (; preconditioner = pc, single_preconditioner = spc, double_budget_tol = dbt, single_rounding_bound = srb, backtracking = bt, presets, single_obs_tol = sot)
end
