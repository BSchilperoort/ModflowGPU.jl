# Multi-model solutions and GWF-GWF exchanges (T6.6, src/gwf/exchange.jl, src/solution.jl):
#
# 1. Universal residual on the merged model of each Tier 7 example (ex-gwf-u1gwfgwf-s1 to
#    -s4, ex-gwf-lgrv-lgr, ex-gwf-spbc): MF6's converged heads (every model's head file)
#    through `formulate!` give a residual below 1e-8 of the largest row magnitude in
#    `:double` (`:single`: 4 eps of each row's rounding scale beyond that); the flows of
#    each model (`model_flowja`) and of each exchange (the IMETH 6 lists of the budget
#    files) with these heads match MF6's within 1e-10 (`:double`) or 1e-4 (`:single`)
#    of the largest.
# 2. Oracle: libmf6's solution matrix and right-hand side after the first `solve`
#    (`SLN_1/AMAT`, `SLN_1/RHS`; models after each other, the exchange connections
#    added, with XT3D on an exchange the interface models' rows) against the merged
#    model's, entry by entry: `:double` < 1e-12, `:single` < 1e-6 relative.
# 3. Unit tests: exchange conductances (`calc_cond_sat`), MF6's input errors of
#    `validate_exchange`, unsupported combinations, and the stepping addresses of several
#    models.
using ModflowGPU: load_simulation, solution_setup, GwfModel, apply_period!, start_timestep!, formulate!, calc_flows!,
    model_views, model_flowja, open_head_file, read_head, open_budget_file, read_budget, find_budget_records, time_steps,
    time_step_window, backend_name, MF6InputError, MF6UnsupportedError

const _EXG_EXAMPLES = ("ex-gwf-u1gwfgwf-s1", "ex-gwf-u1gwfgwf-s2", "ex-gwf-u1gwfgwf-s3", "ex-gwf-u1gwfgwf-s4", "ex-gwf-lgrv-lgr", "ex-gwf-spbc")

# The merged model with MF6's heads of the last saved step (all Tier 7 examples are one
# steady-state step), formulated, flows computed.
function _exchange_state(T, sim, rundir; backend = TEST_BACKEND)
    m = GwfModel(backend, T, solution_setup(sim), ModflowGPU.nper(sim.tdis))
    steps = time_steps(sim.tdis)
    st = last(steps)
    for kper in 1:st.kper
        apply_period!(m, kper; window = time_step_window(steps, findfirst(s -> s.kper == kper, steps)))
    end
    start_timestep!(m, st.delt)
    habs = zeros(Int(m.conn.nodes))
    for v in model_views(m)
        spec = ModflowGPU._vspec(v)
        hf = open_head_file(joinpath(rundir, spec.oc.head_file))
        habs[ModflowGPU._vrange(v)] .= vec(read_head(hf, length(hf)))[v.part.conn.nodeuser]
    end
    copyto!(m.h, T.(habs .- m.setup.r))
    copyto!(m.hold, T.(habs .- m.setup.r))
    scale = KernelAbstractions.zeros(backend, T, m.conn.nodes)
    formulate!(m; scale)
    calc_flows!(m)
    KernelAbstractions.synchronize(backend)
    return m, scale
end

function _exchange_residual(name, rundir)
    sim = load_simulation(rundir)
    return for_each_precision() do p
        T = p.T
        m, scale = _exchange_state(T, sim, rundir)
        conn = m.conn
        res, sc = Float64.(Array(m.res)), Float64.(Array(scale))
        rel = maximum(abs, res) / maximum(sc)
        # `:single`: the rounding of the offset heads and reference differences through
        # the connection terms (XT3D terms, or condsat), plus the boundary terms.
        hoff = Float64.(Array(m.h))
        rb = abs.(Float64.(Array(m.sys.brhs))) .+ abs.(Float64.(Array(m.sys.bdiag)) .* hoff)
        x = m.solution.xt3d
        for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            j = conn.jas[ii]
            if isnothing(x)
                rb[n] += m.setup.condsat[j] * (abs(hoff[n]) + abs(hoff[conn.ja[ii]]) + abs(m.setup.Δr[j]))
            else
                for t in x.tptr[j]:(x.tptr[j + 1] - 1)
                    rb[n] += abs(x.tc[t]) * (abs(hoff[x.ta[t]]) + abs(hoff[x.tb[t]]) + abs(x.tdr[t]))
                end
            end
        end
        excess = maximum((abs.(res) .- 1.0e-8 * maximum(sc)) ./ (rb .+ floatmin()))
        @test T === Float64 ? rel < 1.0e-8 : excess <= 4 * eps(T)
        # Flows of each model and of each exchange against MF6's budget files: in `:single`
        # beyond the rounding of each flow's terms (4 eps(T) of Σ|c|·(|h'_a| + |h'_b| + |Δr|);
        # ex-gwf-lgrv-lgr: K = 1e-6 m/s next to 300 m heads).
        ftol = T === Float64 ? 1.0e-10 : 1.0e-4
        qb = zeros(conn.nja)
        if T === Float32
            for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
                j = conn.jas[ii]
                qb[ii] = isnothing(x) ? m.setup.condsat[j] * (abs(hoff[n]) + abs(hoff[conn.ja[ii]]) + abs(m.setup.Δr[j])) :
                    sum(abs(x.tc[t]) * (abs(hoff[x.ta[t]]) + abs(hoff[x.tb[t]]) + abs(x.tdr[t])) for t in x.tptr[j]:(x.tptr[j + 1] - 1))
            end
            qb .*= 4 * eps(T)
        end
        dq = dx = 0.0
        xf = ModflowGPU.ExchangeFlows(m)
        ModflowGPU.exchange_flows!(xf, m)
        for v in model_views(m)
            spec = ModflowGPU._vspec(v)
            isempty(spec.oc.budget_file) && continue
            bf = open_budget_file(joinpath(rundir, spec.oc.budget_file))
            fj = vec(read_budget(bf, last(find_budget_records(bf; text = "FLOW-JA-FACE", paknam2 = ""))))
            ours = model_flowja(v)
            c = v.part.conn
            offd = [ii for n in 1:c.nodes for ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)]
            dq = max(dq, maximum(max.(abs.(ours[offd] .- fj[offd]) .- qb[v.part.jamap[offd]], 0.0)) / maximum(abs, fj[offd]))
            for (ie, ex) in enumerate(m.solution.exchanges)
                ex.spec.save_flows || continue
                recs = find_budget_records(bf; text = "FLOW-JA-FACE", paknam2 = ex.name)
                isempty(recs) && continue
                mf = read_budget(bf, first(recs)).q
                q = collect(ModflowGPU.exchange_simvals(xf, ie))
                ours_q = ex.m1 == v.index ? q : -q
                dx = max(dx, maximum(max.(abs.(ours_q .- mf) .- qb[ex.pos], 0.0)) / maximum(abs, mf))
            end
        end
        @test dq < ftol
        @test dx < ftol
        @info "Exchange universal residual: $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $(conn.nodes) nodes in $(length(m.solution.models)) model(s), $(length(m.solution.exchanges)) exchange(s)$(isnothing(m.solution.xt3d) ? "" : ", XT3D"): max |res| / max Σ|terms| = $(round(rel; sigdigits = 3)), FLOW-JA-FACE vs MF6 $(round(dq; sigdigits = 3)), exchange flows vs MF6 $(round(dx; sigdigits = 3))"
    end
end

# Oracle: libmf6's SLN_1 system after the first solve against the merged model's, as maps
# (row, column) => value (the structures differ: XT3D stencils, explicit zeros).
function _exchange_oracle(name, simdir)
    sim = load_simulation(simdir)
    o = _libmf6_first_solve(simdir)
    sol = solution_setup(sim)
    @test length(o.ia) == sol.conn.nodes + 1
    r = sol.setup.r
    mf = Dict{Tuple{Int, Int}, Float64}()
    bmf = copy(o.rhs)
    bscale = abs.(o.rhs)
    for n in 1:sol.conn.nodes, ii in o.ia[n]:(o.ia[n + 1] - 1)
        mf[(n, Int(o.ja[ii]))] = o.amat[ii]
        t = o.amat[ii] * r[o.ja[ii]]
        bmf[n] -= t
        bscale[n] += abs(t)
    end
    amax = maximum(abs, o.amat)
    for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, sol, ModflowGPU.nper(sim.tdis))
        apply_period!(m, 1; window = time_step_window(time_steps(sim.tdis), 1))
        start_timestep!(m, o.delt)
        # CG: the constant-head columns moved to the right-hand side, as MF6 (sln_ls).
        formulate!(m; symmetric = only(only(sim.solution_groups).solutions).linear_acceleration === :cg)
        KernelAbstractions.synchronize(TEST_BACKEND)
        @test maximum(abs.(r .+ Float64.(Array(m.h)) .- o.x0)) <= 4 * eps(T) * maximum(abs, o.x0)
        ia, ja, a = Array(m.scsr.ia), Array(m.scsr.ja), Float64.(Array(m.sys.amat))
        ours = Dict{Tuple{Int, Int}, Float64}()
        for n in 1:sol.conn.nodes, ii in ia[n]:(ia[n + 1] - 1)
            ours[(n, Int(ja[ii]))] = a[ii]
        end
        ea = maximum(k -> abs(get(ours, k, 0.0) - get(mf, k, 0.0)), union(keys(ours), keys(mf))) / amax
        eb = maximum(abs.(Float64.(Array(m.sys.rhs)) .- bmf)) / maximum(bscale)
        tol = T === Float64 ? 1.0e-12 : 1.0e-6
        @test ea < tol
        @test eb < tol
        @info "Exchange oracle vs libmf6 (first solve): $name, :$(p.preset): amat max |Δ| / max |a| = $(round(ea; sigdigits = 3)), b' max |Δ| / max(|RHS| + Σ|a·r|) = $(round(eb; sigdigits = 3)) ($(length(mf)) MF6 entries, $(length(ours)) ours)"
    end
    return nothing
end

@testset "GWF-GWF exchanges: universal residual and libmf6 oracle (Tier 7)" begin
    for name in _EXG_EXAMPLES
        test_model_selected(name) || continue
        rundir = reference_rundir(name)
        if isnothing(rundir)
            @info "Skipping $name in the exchange tests: no converged reference run in $(joinpath(REFERENCE_DIR, "runs", name))"
            continue
        end
        @testset "$name" begin
            _exchange_residual(name, rundir)
            if libmf6_available()
                _exchange_oracle(name, joinpath(REFERENCE_DIR, "mf6examples", name))
            else
                @info "Skipping the libmf6 oracle of $name: libmf6 not available"
            end
        end
    end
end

# --- Unit tests on a small simulation written here: two 1-layer DIS models side by side
# (left: 3×4, right: 3×4), connected along their shared edge by an exchange, CHD at the
# outer columns.
function _write_two_models(dir; exg_options = "", aux = true, k22 = false, xt3d_npf = false, solutions = 1, icelltype = 0, extra_exchange = "", obs = false)
    mkpath(dir)
    nam(m) = """
    BEGIN options
    END options
    BEGIN packages
      DIS6  $m.dis  DIS
      NPF6  $m.npf  NPF
      IC6  $m.ic  IC
      CHD6  $m.chd  CHD-$(uppercase(m))
      OC6  $m.oc  OC
    END packages
    """
    for (m, xo, h) in (("left", 0.0, 10.0), ("right", 400.0, 0.0))
        write(joinpath(dir, "$m.nam"), nam(m))
        write(
            joinpath(dir, "$m.dis"), """
            BEGIN options
              XORIGIN $xo
            END options
            BEGIN dimensions
              NLAY 1
              NROW 3
              NCOL 4
            END dimensions
            BEGIN griddata
              delr
                CONSTANT 100.0
              delc
                CONSTANT 100.0
              top
                CONSTANT 0.0
              botm
                CONSTANT -10.0
            END griddata
            """,
        )
        write(
            joinpath(dir, "$m.npf"), """
            BEGIN options
            $(xt3d_npf ? "  XT3D" : "")
            END options
            BEGIN griddata
              icelltype
                CONSTANT $icelltype
              k
                CONSTANT $(m == "left" ? 2.0 : 5.0)
            $(k22 ? "  k22\n    CONSTANT 1.0" : "")
            END griddata
            """,
        )
        write(joinpath(dir, "$m.ic"), "BEGIN griddata\n  strt\n    CONSTANT 5.0\nEND griddata\n")
        col = m == "left" ? 1 : 4
        write(
            joinpath(dir, "$m.chd"), """
            BEGIN dimensions
              MAXBOUND 3
            END dimensions
            BEGIN period 1
              1 1 $col $h
              1 2 $col $h
              1 3 $col $h
            END period 1
            """,
        )
        write(joinpath(dir, "$m.oc"), "BEGIN options\n  HEAD FILEOUT $m.hds\n  BUDGET FILEOUT $m.cbc\nEND options\nBEGIN period 1\n  SAVE HEAD ALL\n  SAVE BUDGET ALL\nEND period 1\n")
    end
    auxline = aux ? "  AUXILIARY ANGLDEGX CDIST\n" : ""
    bnames = ("north", "south", "south")
    rows = join(["  1 $i 4  1 $i 1  1  50.0  50.0  100.0" * (aux ? "  0.0  100.0" : "") * (obs ? "  $(bnames[i])" : "") for i in 1:3], "\n")
    if obs
        exg_options *= "\n  BOUNDNAMES\n  OBS6 FILEIN sim.gwfgwf.obs"
        write(
            joinpath(dir, "sim.gwfgwf.obs"), """
            BEGIN continuous FILEOUT sim.gwfgwf.obs.csv
              q2 FLOW-JA-FACE 2
              qsouth FLOW-JA-FACE south
            END continuous
            """,
        )
    end
    write(
        joinpath(dir, "sim.gwfgwf"), """
        BEGIN options
        $(auxline)  SAVE_FLOWS
        $exg_options
        END options
        BEGIN dimensions
          NEXG 3
        END dimensions
        BEGIN exchangedata
        $rows
        END exchangedata
        """,
    )
    write(joinpath(dir, "sim.tdis"), "BEGIN dimensions\n  NPER 1\nEND dimensions\nBEGIN perioddata\n  1.0 1 1.0\nEND perioddata\n")
    write(
        joinpath(dir, "sim.ims"), """
        BEGIN nonlinear
          OUTER_DVCLOSE 1.0e-9
          OUTER_MAXIMUM 50
        END nonlinear
        BEGIN linear
          INNER_MAXIMUM 200
          INNER_DVCLOSE 1.0e-10
          INNER_RCLOSE 1.0e-10
          LINEAR_ACCELERATION BICGSTAB
        END linear
        """,
    )
    groups = solutions == 1 ? "BEGIN solutiongroup 1\n  IMS6 sim.ims left right\nEND solutiongroup 1\n" :
        "BEGIN solutiongroup 1\n  MXITER 2\n  IMS6 sim.ims left\n  IMS6 sim.ims right\nEND solutiongroup 1\n"
    write(
        joinpath(dir, "mfsim.nam"), """
        BEGIN options
        END options
        BEGIN timing
          TDIS6 sim.tdis
        END timing
        BEGIN models
          GWF6 left.nam left
          GWF6 right.nam right
        END models
        BEGIN exchanges
          GWF6-GWF6 sim.gwfgwf left right
        $extra_exchange
        END exchanges
        $groups
        """,
    )
    return dir
end

@testset "GWF-GWF exchanges: conductances, errors and addresses" begin
    mktempdir() do tmp
        dir = _write_two_models(joinpath(tmp, "two"))
        sim = load_simulation(dir)
        sol = solution_setup(sim)
        ex = only(sol.exchanges)
        @test ex.name == "GWF-GWF_1"
        @test (ex.m1, ex.m2) == (1, 2)
        @test ex.g1 == Int32[4, 8, 12] && ex.g2 == Int32[13, 17, 21]
        # calc_cond_sat, harmonic: hwva·T1·T2 / (T1·cl2 + T2·cl1) with T = K·thickness.
        t1, t2 = 2.0 * 10.0, 5.0 * 10.0
        @test ex.condsat ≈ fill(100.0 * t1 * t2 / (t1 * 50.0 + t2 * 50.0), 3) rtol = 1.0e-14
        @test count(sol.exgentry) == 6
        @test sol.conn.nodes == 24 && sol.conn.njas == 2 * 17 + 3
        # Logarithmic averaging of the exchange.
        sol2 = solution_setup(load_simulation(_write_two_models(joinpath(tmp, "log"); exg_options = "  CELL_AVERAGING LOGARITHMIC")))
        @test only(sol2.exchanges).condsat ≈ fill(ModflowGPU.condmean(2.0, 5.0, 10.0, 10.0, 50.0, 50.0, 100.0, ModflowGPU.CCOND_LMEAN), 3) rtol = 1.0e-14
        # A run: both models' files, the exchange terms in each model's budget.
        out = joinpath(tmp, "out")
        r = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = :double, outdir = out)
        @test r.converged
        @test [x.name for x in r.models] == ["LEFT", "RIGHT"]
        @test r.models[2].nodes == 13:24
        @test all(isfile, (r.models[1].head_file, r.models[2].budget_file, r.models[2].listing_file))
        bl, br = r.budgets
        @test (bl.model, br.model) == ("LEFT", "RIGHT")
        @test [t.name for t in bl.terms] == ["CHD", "FLOW-JA-FACE", "FLOW-JA-FACE-CHD"]
        q = only(t for t in bl.terms if t.name == "FLOW-JA-FACE").rate_out
        @test q > 0
        @test only(t for t in br.terms if t.name == "FLOW-JA-FACE").rate_in ≈ q rtol = 1.0e-12
        @test abs(ModflowGPU.percent_discrepancy(bl)) < 1.0e-8 && abs(ModflowGPU.percent_discrepancy(br)) < 1.0e-8
        # Linear heads: the flow through the two models is the same conductance chain.
        bf = open_budget_file(r.models[2].budget_file)
        lst = read_budget(bf, only(find_budget_records(bf; text = "FLOW-JA-FACE", paknam2 = "GWF-GWF_1")))
        @test lst.id1 == Int32[1, 5, 9] && lst.id2 == Int32[4, 8, 12]
        @test sum(lst.q) ≈ q rtol = 1.0e-12
        @test lst.aux == repeat([0.0, 100.0], 1, 3)

        # Stepping addresses of several models.
        s = initialize(dir; backend = TEST_BACKEND, precision = :double, write_output = false)
        a = var_addresses(s)
        @test "LEFT/X" in a && "RIGHT/X" in a && "RIGHT/CHD-RIGHT/BOUND" in a && "LEFT/DIS/NODES" in a
        @test s.models == ["LEFT", "RIGHT"]
        @test length(get_value(s, "RIGHT/X")) == 12
        ModflowGPU.update(s)
        hr = get_value(s, "RIGHT/X")
        @test hr ≈ r.heads[13:24] rtol = 1.0e-12
        @test get_value(s, "RIGHT/CHD-RIGHT/NODELIST") == Int32[4, 8, 12]
        @test only(get_value(s, "RIGHT/DIS/NODES")) == 12
        x = get_value_ptr(s, "LEFT/X")
        @test x ≈ r.heads[1:12] rtol = 1.0e-12
        set_value(s, "RIGHT/X", fill(1.0, 12))
        @test get_value(s, "RIGHT/X") ≈ fill(1.0, 12)
        @test get_value(s, "LEFT/X") ≈ r.heads[1:12] rtol = 1.0e-12
        @test length(s.budgets) == 2
        # BMI: model 2's grids follow model 1's (1, 2, then 3, 4 for its CHD package).
        @test ModflowGPU.BMI.get_var_grid(s, "LEFT/X") == 1 && ModflowGPU.BMI.get_var_grid(s, "RIGHT/X") == 5
        @test ModflowGPU.BMI.get_var_grid(s, "RIGHT/CHD-RIGHT/BOUND") == 8
        @test ModflowGPU.BMI.get_grid_size(s, 5) == 12 && ModflowGPU.BMI.get_grid_size(s, 8) == 3
        gx = zeros(4)
        ModflowGPU.BMI.get_grid_x(s, 5, gx)
        @test gx ≈ [450.0, 550.0, 650.0, 750.0]
        @test ModflowGPU.BMI.get_var_units(s, "RIGHT/X") == "L"
        finalize_simulation(s)
        # libmf6's BMI (MF6Model): the grid of a model is its number.
        mm = MF6Model(dir; backend = TEST_BACKEND, precision = :double, write_output = false)
        ModflowGPU.BMI.initialize(mm)
        @test ModflowGPU.BMI.get_var_grid(mm, "LEFT/X") == 1 && ModflowGPU.BMI.get_var_grid(mm, "RIGHT/X") == 2
        @test ModflowGPU.BMI.get_grid_shape(mm, 2, zeros(Int32, 3))[1:2] == Int32[3, 4]
        ModflowGPU.BMI.finalize(mm)

        # MF6's input errors (exg-gwfgwf.f90 validate_exchange) and what is not supported.
        e = try
            solution_setup(load_simulation(_write_two_models(joinpath(tmp, "k22"); aux = false, k22 = true)))
        catch err
            err
        end
        @test e isa MF6InputError && occursin("because K22 was specified", e.msg)
        e = try
            solution_setup(load_simulation(_write_two_models(joinpath(tmp, "x3"); aux = false, exg_options = "  XT3D")))
        catch err
            err
        end
        @test e isa MF6InputError && occursin("because XT3D is enabled", e.msg)
        e = try
            initialize(_write_two_models(joinpath(tmp, "conv"); icelltype = 1); write_output = false)
        catch err
            err
        end
        @test e isa MF6UnsupportedError && occursin("convertible cells", e.msg)
        e = try
            load_simulation(_write_two_models(joinpath(tmp, "two_sol"); solutions = 2))
        catch err
            err
        end
        @test e isa MF6UnsupportedError && occursin("more than one solution", e.msg)
        e = try
            load_simulation(_write_two_models(joinpath(tmp, "gnc"); exg_options = "  GNC6 FILEIN sim.gnc"))
        catch err
            err
        end
        @test e isa MF6UnsupportedError

        # Against mf6 (run at test time): heads, budgets and listing tables of both models
        # (`_run_parity`), and the exchange observations (by number and by boundary name).
        if isfile(RUN_MF6)
            odir = _write_two_models(joinpath(tmp, "obs"); obs = true)
            out = read(Cmd(`$RUN_MF6`; dir = odir), String)
            @test occursin("Normal termination", out)
            _run_parity("two models (exchange test)", odir)
            mktempdir() do o2
                r = ModflowGPU.run(odir; backend = TEST_BACKEND, precision = :double, outdir = o2)
                f = only(filter(endswith("sim.gwfgwf.obs.csv"), r.obs_files))
                a, b = ModflowGPU.read_observations(f), ModflowGPU.read_observations(joinpath(odir, "sim.gwfgwf.obs.csv"))
                @test a.names == b.names == ["Q2", "QSOUTH"]
                @test a.times == b.times
                @test a.values ≈ b.values rtol = 1.0e-10
                @test b.values[1, 2] ≈ 2 * b.values[1, 1] rtol = 1.0e-6
            end
            # gwf_gwf_rp_obs: an exchange number beyond NEXG.
            bad = _write_two_models(joinpath(tmp, "obsbad"); obs = true)
            write(joinpath(bad, "sim.gwfgwf.obs"), "BEGIN continuous FILEOUT x.csv\n  q7 FLOW-JA-FACE 7\nEND continuous\n")
            e = input_error_of(() -> load_simulation(bad))
            @test occursin("Exchange id \"7\" for observation \"FLOW-JA-FACE\" is invalid in package \"GWF-GWF_1\"", e.msg)
        else
            @info "Skipping the exchange test against mf6: mf6 not found at $RUN_MF6"
        end
    end
end
