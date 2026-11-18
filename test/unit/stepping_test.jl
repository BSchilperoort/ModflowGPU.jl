# Stepping API and BMI (T3.3): test/models/coupling_rde driven through the call sequence of
# ModflowCoupling's `advance!` (reference/Modflow-coupling/src/model.jl), against
# `ModflowGPU.run`, bitwise:
#
# - without changes: heads, SIMVALS and budgets equal those of `run` at every step;
# - with boundary values set between steps (RIV stage and conductance written into the
#   BOUND host mirror, as ModflowCoupling does; RCHA recharge through `set_value`) at the
#   start of stress period 2: equal to `run` on a copy of the model whose input has those
#   values as a PERIOD 2 block (so set values go through the same MULTIPLIER and offset
#   conversion as period data).
const BMI = ModflowGPU.BMI  # BasicModelInterface, through the package (not a test dependency)
using ModflowGPU: Simulation, initialize, prepare_time_step, prepare_solve, solve, finalize_solve,
    finalize_time_step, get_value_ptr, get_value, set_value, get_var_address, get_current_time, get_end_time,
    get_time_step

const STEPPING_MODEL = joinpath(@__DIR__, "..", "models", "coupling_rde")

# A copy of coupling_rde with RIV and RCHA PERIOD 2 blocks: the period-3 stages, 1.5 × the
# conductances (before the cmult multiplier) and recharge `rch`.
function _stepping_variant(dir, stages, conds, rch)
    cp(STEPPING_MODEL, dir; force = true)
    riv = joinpath(dir, "rde.riv")
    lines = readlines(riv)
    p1 = findfirst(==("BEGIN period  1"), lines)
    e1 = findfirst(==("END period  1"), lines)
    block = ["", "BEGIN period  2"]
    for (i, l) in enumerate(lines[(p1 + 1):(e1 - 1)])
        w = split(l)
        push!(block, "  $(w[1]) $(w[2]) $(w[3])  $(repr(stages[i]))  $(repr(conds[i]))  $(w[6])  $(w[7])  $(w[8])")
    end
    push!(block, "END period  2")
    write(riv, join([lines[1:e1]; block; lines[(e1 + 1):end]], "\n") * "\n")
    rcha = joinpath(dir, "rde.rcha")
    txt = read(rcha, String)
    write(rcha, replace(txt, "BEGIN period  3" => "BEGIN period  2\n  recharge\n    CONSTANT  $(repr(rch))\nEND period  2\n\nBEGIN period  3"))
    return dir
end

# ModflowCoupling's `advance!` for every period (one step per call here: the model has
# several steps per period, each driven the same way). `stress!(ptr, step)` writes the
# BOUND mirrors after `prepare_time_step`. Returns per step the heads and SIMVALS read
# through the pointers after `finalize_time_step`.
function _drive_like_coupling(s::Simulation, stress!)
    mname = s.model.spec.name
    addr(var, sub = "") = get_var_address(s, var, mname; subcomponent_name = sub)
    ptr = (
        head = get_value_ptr(s, addr("X")),
        nodeuser = Int.(vec(get_value_ptr(s, addr("NODEUSER", "DIS")))),
        rch_bound = get_value_ptr(s, addr("BOUND", "RCHA")),
        riv_bound = get_value_ptr(s, addr("BOUND", "RIV")),
        riv_simvals = get_value_ptr(s, addr("SIMVALS", "RIV")),
        drn_simvals = get_value_ptr(s, addr("SIMVALS", "DRN")),
        evt_simvals = get_value_ptr(s, addr("SIMVALS", "EVT")),
        mxiter = Int(only(get_value_ptr(s, get_var_address(s, "MXITER", "SLN_1")))),
    )
    out = []
    k = 0
    while get_current_time(s) < get_end_time(s)
        prepare_time_step(s, 0.0)
        k += 1
        if k == 1  # calibrate_bmi!: record order and the stored placeholder RCH rate
            for name in ("RIV", "DRN")
                nodelist = vec(get_value_ptr(s, addr("NODELIST", name)))
                pkg = s.model.packages[findfirst(p -> p.spec.name == name, s.model.packages)]
                users = [ptr.nodeuser[n] for n in nodelist[1:pkg.nbound]]
                @test users == pkg.spec.periods.data[1].nodeu
            end
        end
        stress!(ptr, s.steps[s.istep])
        prepare_solve(s)
        converged = false
        for _ in 1:ptr.mxiter
            converged = solve(s)
            converged && break
        end
        finalize_solve(s)
        finalize_time_step(s)
        @test converged
        push!(out, (; head = copy(ptr.head), riv = copy(ptr.riv_simvals), drn = copy(ptr.drn_simvals), evt = copy(ptr.evt_simvals)))
    end
    return out
end

_budget_rates(b) = [(t.name, t.package, t.rate_in, t.rate_out, t.cum_in, t.cum_out) for t in b.terms]

@testset "stepping" begin
    for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
        kw = (; backend = TEST_BACKEND, precision = p.preset, write_output = false)
        ref = ModflowGPU.run(STEPPING_MODEL; kw...)

        @testset "advance! sequence, no changes: bitwise equal to run" begin
            s = initialize(STEPPING_MODEL; kw...)
            out = _drive_like_coupling(s, (ptr, step) -> nothing)
            @test length(out) == length(ref.steps) == 10
            @test last(out).head == ref.heads
            @test [_budget_rates(b) for b in s.budgets] == [_budget_rates(b) for b in ref.budgets]
            @test [r.outer.iterations for r in s.results] == [r.outer.iterations for r in ref.steps]
            @test get_value(s, "RDE/X") == ref.heads
            @test vec(get_value(s, "RDE/RIV/SIMVALS")) == last(out).riv
            ModflowGPU.finalize(s)
        end

        @testset "boundary values set between steps: bitwise equal to run with PERIOD 2 data" begin
            spec = ModflowGPU.load_simulation(STEPPING_MODEL)
            rivspec = only(b for b in only(spec.models).boundaries if b.name == "RIV")
            per1, per3 = rivspec.periods.data[1], rivspec.periods.data[2]  # PERIOD blocks 1 and 3
            rows = ModflowGPU.list_column(rivspec.schema, "stage").first, ModflowGPU.list_column(rivspec.schema, "cond").first
            stages = Vector{Float64}(per3.bound[rows[1], :])
            conds = 1.5 .* per1.bound[rows[2], :]
            rch = 0.0007
            dir = _stepping_variant(mktempdir(), stages, conds, rch)
            ref2 = ModflowGPU.run(dir; kw...)
            @test ref2.heads != ref.heads

            s = initialize(STEPPING_MODEL; kw...)
            out = _drive_like_coupling(
                s, function (ptr, step)
                    step.kper == 2 && step.kstp == 1 || return
                    ptr.riv_bound[1, :] .= stages
                    ptr.riv_bound[2, :] .= conds
                    set_value(s, "RDE/RCHA/BOUND", fill(rch, size(ptr.rch_bound)))
                    @test ptr.rch_bound == fill(rch, size(ptr.rch_bound))
                end
            )
            @test last(out).head == ref2.heads
            @test [_budget_rates(b) for b in s.budgets] == [_budget_rates(b) for b in ref2.budgets]
            # Period 3 data replaces the set values (as in MF6); the multiplier applies to them.
            riv = get_value_ptr(s, "RDE/RIV/BOUND")
            @test riv[1, :] == stages && riv[2, :] == per3.bound[rows[2], :]
            ModflowGPU.finalize(s)

            # T5.2 (ModflowCoupling's device-resident coupling): the same values written
            # straight into the device BOUND arrays (`get_value_ptr(...; device = true)`:
            # offsets, multiplier applied) between prepare_time_step and prepare_solve take
            # effect as the mirror writes do, also with host mirrors of BOUND present (they
            # are not dirty, so prepare_solve sends nothing over the device values).
            # Captured from a run with mirror writes, in period 2 after the conversion.
            dev(s, pkg) = get_value_ptr(s, "RDE/$pkg/BOUND"; device = true)
            captured = Dict{String, Any}()
            s = initialize(STEPPING_MODEL; kw...)
            _drive_like_coupling(
                s, function (ptr, step)
                    if step.kper == 2 && step.kstp == 1
                        ptr.riv_bound[1, :] .= stages
                        ptr.riv_bound[2, :] .= conds
                        set_value(s, "RDE/RCHA/BOUND", fill(rch, size(ptr.rch_bound)))
                    elseif step.kper == 2 && step.kstp == 2
                        captured["RIV"], captured["RCHA"] = Array(dev(s, "RIV")), Array(dev(s, "RCHA"))
                    end
                end
            )
            ModflowGPU.finalize(s)
            # T5.4: written through `device_bound!` (the enforced contract: checked at
            # prepare_solve), the same arrays.
            s = initialize(STEPPING_MODEL; kw...)
            out = _drive_like_coupling(
                s, function (ptr, step)
                    step.kper == 2 && step.kstp == 1 || return
                    @test eltype(dev(s, "RIV")) === p.T
                    @test ModflowGPU.device_bound!(s, "RDE/RIV/BOUND") === dev(s, "RIV")
                    copyto!(ModflowGPU.device_bound!(s, "RDE/RIV/BOUND"), captured["RIV"])
                    copyto!(ModflowGPU.device_bound!(s, "RDE/RCHA/BOUND"), captured["RCHA"])
                end
            )
            @test last(out).head == ref2.heads
            @test [_budget_rates(b) for b in s.budgets] == [_budget_rates(b) for b in ref2.budgets]
            ModflowGPU.finalize(s)
        end

        @testset "API checks" begin
            s = initialize(STEPPING_MODEL; kw...)
            @test get_current_time(s) == 0.0 && get_end_time(s) == 51.0
            @test_throws ArgumentError solve(s)
            @test_throws ArgumentError set_value(s, "RDE/RIV/SIMVALS", zeros(12))
            @test_throws ArgumentError get_value(s, "RDE/NOPE")
            @test get_value(s, "RDE/X") ≈ get_value(s, "RDE/HREF") .+ Float64.(Array(get_value_ptr(s, "RDE/X"; device = true)))
            @test get_value_ptr(s, "RDE/X"; device = true) === s.model.h
            prepare_time_step(s)
            @test get_time_step(s) == 1.0 && only(get_value(s, "TDIS/KPER")) == 1
            @test_throws ArgumentError prepare_time_step(s)
            # X written through the mirror before prepare_solve becomes the starting head.
            x = get_value_ptr(s, "RDE/X")
            x .+= 0.5
            prepare_solve(s)
            @test get_value(s, "RDE/XOLD") ≈ x
            # A write through the mirror after prepare_solve is discarded, with a warning.
            x[1] += 1.0
            @test_logs (:warn, r"discarded") match_mode = :any finalize_solve(s)
            ModflowGPU.finalize(s)

            # T5.4: device BOUND writes only between prepare_time_step and prepare_solve,
            # checked at prepare_solve (riv_ck on the device: stage >= bottom, finite).
            s = initialize(STEPPING_MODEL; kw...)
            @test_throws ArgumentError ModflowGPU.device_bound!(s, "RDE/RIV/BOUND")
            prepare_time_step(s)
            @test_throws ArgumentError ModflowGPU.device_bound!(s, "RDE/RIV/SIMVALS")
            riv = ModflowGPU.device_bound!(s, "RDE/RIV/BOUND")
            good = Array(riv)
            bad = copy(good)
            bad[1, 2] = bad[3, 2] - p.T(1)   # stage below the river bottom
            copyto!(riv, bad)
            @test_throws r"RIV.*record 2 written on the device: RIVER STAGE IS LESS THAN RIVER BOTTOM" prepare_solve(s)
            bad = copy(good)
            bad[2, 5] = p.T(NaN)
            copyto!(riv, bad)
            @test_throws r"record 5 written on the device: non-finite" prepare_solve(s)
            copyto!(riv, good)
            prepare_solve(s)   # the check passes; the window is closed
            @test_throws ArgumentError ModflowGPU.device_bound!(s, "RDE/RIV/BOUND")
            finalize_solve(s)
            ModflowGPU.finalize(s)
            @test_throws ArgumentError get_value(s, "RDE/X")
            @test_throws ArgumentError get_value_ptr(s, "RDE/X"; device = true)

            # BMI: time, grid, variables, update until the end.
            # (T8.9: settings from a configuration file; test/unit/bmi_test.jl checks the rest.)
            cfg = joinpath(mktempdir(), "bmi.toml")
            write(cfg, "simulation = \"$(escape_string(abspath(STEPPING_MODEL)))\"\nbackend = \"$TEST_BACKEND_NAME\"\nprecision = \"$(p.preset)\"\nwrite_output = false\n")
            b = BMI.initialize(Simulation, cfg)
            @test BMI.get_time_units(b) == "days"
            @test BMI.get_grid_rank(b, 1) == 3 && BMI.get_grid_shape(b, 1, zeros(Int, 3)) == [2, 12, 15]
            # 337 of 360 cells active: the heads are on grid 2 (the model nodes)
            @test BMI.get_grid_size(b, 1) == 360 && BMI.get_var_grid(b, "RDE/X") == 2 && BMI.get_grid_size(b, 2) == 337
            @test BMI.get_var_type(b, "RDE/X") == "Float64" && BMI.get_var_type(b, "RDE/RIV/NODELIST") == "Int32"
            @test BMI.get_var_nbytes(b, "RDE/RIV/BOUND") == 3 * 12 * 8
            @test "RDE/RIV/BOUND" in BMI.get_input_var_names(b) && !("RDE/RIV/SIMVALS" in BMI.get_input_var_names(b))
            BMI.update(b)
            @test BMI.get_current_time(b) == 1.0
            BMI.update_until(b, BMI.get_end_time(b))
            h = zeros(length(ref.heads))
            BMI.get_value(b, "RDE/X", h)
            @test h == ref.heads
            v = zeros(2)
            BMI.get_value_at_indices(b, "RDE/X", v, [1, 5])
            @test v == ref.heads[[1, 5]]
            BMI.finalize(b)
        end
    end
end

# T5.4: `finalize` frees the device arrays of the model, solver and budget accumulator, so
# the pool usage of a GPU backend returns to its level before `initialize` (ROCm:
# `AMDGPU.used_memory`, the stream-ordered pool, measured without a GC in between). On
# the CPU backend nothing is freed (host arrays).
# T6.2 (from T6.1): no output into REFERENCE_DIR (the default `outdir` is the input
# directory, which once overwrote a reference run).
@testset "no output into REFERENCE_DIR" begin
    mktempdir() do tmp
        ref = joinpath(tmp, "reference")
        dir = joinpath(ref, "runs", "rde")
        mkpath(dirname(dir))
        cp(STEPPING_MODEL, dir)
        withenv("REFERENCE_DIR" => ref) do
            @test_throws r"refusing to write output into .*REFERENCE_DIR" ModflowGPU.run(dir; backend = TEST_BACKEND)
            @test_throws ArgumentError initialize(dir; backend = TEST_BACKEND, outdir = joinpath(ref, "out"))
            @test !isfile(joinpath(dir, "rde.hds"))
            # Elsewhere, or without output: fine.
            @test ModflowGPU.run(dir; backend = TEST_BACKEND, outdir = joinpath(tmp, "out")).converged
            @test ModflowGPU.run(dir; backend = TEST_BACKEND, write_output = false).converged
        end
    end
end

@testset "finalize frees device memory" begin
    pool_used() = TEST_BACKEND_NAME == "rocm" ? (AMDGPU.synchronize(); AMDGPU.used_memory()) : nothing
    for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
        kw = (; backend = TEST_BACKEND, precision = p.preset, write_output = false)
        for preconditioner in (:mc_ilu0, :amg)
            GC.gc(true)
            before = pool_used()
            s = initialize(STEPPING_MODEL; kw..., preconditioner)
            for _ in 1:3
                ModflowGPU.update(s)
            end
            during = pool_used()
            nfreed = ModflowGPU.free_device_arrays!(nothing)   # nothing to free
            @test nfreed == 0
            ModflowGPU.finalize(s)
            after = pool_used()
            @test s.stage === :freed
            if TEST_BACKEND isa CPU
                @test ModflowGPU.free_device_arrays!(s.model) == 0
            else
                @info "device pool (bytes) for $(p.preset), $preconditioner" before during after
                @test during > before
                @test after == before
            end
            ModflowGPU.finalize(s)   # a second call does nothing
        end
    end
end

# Section timer (T4.1): `@timed_section` hooks are called in nested pairs when a timer is
# installed, and change nothing in the results.
mutable struct _CountingTimer
    stack::Vector{Symbol}
    counts::Dict{Symbol, Int}
end
ModflowGPU.section_begin(t::_CountingTimer, name::Symbol) = push!(t.stack, name)
function ModflowGPU.section_end(t::_CountingTimer, name::Symbol)
    pop!(t.stack) === name || error("unbalanced section $name")
    t.counts[name] = get(t.counts, name, 0) + 1
    return nothing
end

@testset "section timer" begin
    kw = (; backend = TEST_BACKEND, precision = :double, write_output = false)
    @test ModflowGPU.section_timer() === nothing
    ref = ModflowGPU.run(STEPPING_MODEL; kw...)
    t = _CountingTimer(Symbol[], Dict{Symbol, Int}())
    prev = ModflowGPU.set_section_timer!(t)
    r = try
        ModflowGPU.run(STEPPING_MODEL; kw...)
    finally
        ModflowGPU.set_section_timer!(prev)
    end
    @test ModflowGPU.section_timer() === nothing
    @test isempty(t.stack)
    @test r.heads == ref.heads
    nouter = sum(s -> s.outer.iterations, r.steps)
    ninner = sum(s -> s.outer.inner, r.steps)
    @test t.counts[:assembly] == nouter
    @test t.counts[:linear_solve] == nouter
    @test t.counts[:spmv_dot] >= ninner
    @test t.counts[:budget] == length(r.steps)
    @test all(k -> haskey(t.counts, k), (:precond_setup, :update_reduce, :vector_update, :outer_reduction, :period_data, :flows))
end

# The address table of the user manual (docs/src/coupling/stepping.md) lists every address
# of `var_addresses` of a model with every supported stress package (list and array
# input), with `<model>` and `<package>` for the names, and no address that does not exist.
@testset "addresses documented" begin
    dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-10.0, 4, 3, 1))
    cell = [(1, 2, 2)]
    list(kind, values) = build_boundary(kind, dis; periods = [1 => (; cells = cell, values)])
    bs = [
        list(:CHD, [0.0]), list(:WEL, [-1.0]), list(:RCH, [1.0e-3]), list(:GHB, [-1.0 1.0]),
        list(:RIV, [-1.0 1.0 -2.0]), list(:DRN, [-1.0 1.0]), list(:EVT, [0.0 1.0e-3 1.0]),
        build_array_boundary(:RCH, dis; periods = [1 => (; recharge = 1.0e-4)]),
        build_array_boundary(:EVT, dis; periods = [1 => (; surface = 0.0, rate = 1.0e-3, depth = 1.0)]),
    ]
    @test Set(ModflowGPU.boundary_kind.(bs)) == Set(ModflowGPU.BOUNDARY_KINDS)
    gwf = build_model("all"; dis, npf = build_npf(dis; k = 1.0), ic = build_ic(dis; strt = 0.0), boundaries = bs)
    s = initialize(build_simulation(gwf; tdis = build_tdis(; perlen = 1.0)); backend = TEST_BACKEND, write_output = false)
    pkgs = [uppercase(b.name) for b in gwf.boundaries]
    function generic(a)
        a = replace(a, r"^ALL/" => "<model>/")
        for p in pkgs
            a = replace(a, "<model>/$p/" => "<model>/<package>/")
        end
        return a
    end
    addresses = Set(generic.(var_addresses(s)))
    finalize_simulation(s)
    page = read(joinpath(pkgdir(ModflowGPU), "docs", "src", "coupling", "stepping.md"), String)
    documented = Set(m[1] for m in eachmatch(r"`((?:<model>|SLN_1|TDIS)(?:/(?:<package>|[A-Z0-9_-]+))+)`", page))
    @test isempty(setdiff(addresses, documented))   # an address missing from the page
    @test isempty(setdiff(documented, addresses))   # a documented address that does not exist
end
