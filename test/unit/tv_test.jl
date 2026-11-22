# TVK6 and TVS6, time-varying hydraulic conductivity and storage (T6.12, src/gwf/tv.jl and
# the TVS part of src/gwf/sto.jl):
#
# 1. Input: the TVK6/TVS6 files into `NpfSpec.tvk` / `StoSpec.tvs` (PERIOD records, TS6
#    series, DISABLE_STORAGE_CHANGE_INTEGRATION), the reader's errors, write_simulation
#    round trip, and the combinations that are unsupported (XT3D, HFB6, multi-model).
# 2. Engine: MF6's errors when the settings are applied (K <= 0: the autotest tvk02_negk;
#    SY without convertible cells; a cell outside the active domain); the device
#    conductances after a change equal those of NPF with the changed K (only the changed
#    connections written); the storage-change data of a step with a TVS change; settings
#    equal to the NPF/STO values change nothing, bitwise.
# 3. Run parity against mf6 (run at test time) in `:single` and `:double`: the autotests
#    test_gwf_npf_tvk01..06 and test_gwf_sto_tvs01 (test/models/tv, generate.jl there), and
#    test/models/tv/tv_conv (convertible cells, TVK with a time series, TVS of confined and
#    convertible cells) with its NEWTON and DISABLE_STORAGE_CHANGE_INTEGRATION variants.
using ModflowGPU: load_simulation, build_connectivity, npf_setup, GwfModel, apply_period!, advance_time_series!,
    start_timestep!, MF6UnsupportedError, MF6InputError, write_simulation, spec_identical, solution_setup, time_steps,
    time_step_window, TvSpec, TvPeriod, NpfSpec, StoSpec, GwfSpec

const TV_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))
const TV_MODELS = joinpath(@__DIR__, "..", "models", "tv")

# A copy of the simulation `src` (inputs only) in `dst` with `edit(name, text)` applied.
function _tv_variant(src, dst; edit = (f, t) -> t)
    mkpath(dst)
    for f in readdir(src)
        (f == "generate.jl" || any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".grb", ".out"))) && continue
        write(joinpath(dst, f), edit(f, read(joinpath(src, f), String)))
    end
    chmod(dst, 0o755; recursive = true)
    return dst
end

_tv_newton(f, t) = endswith(f, ".nam") && f != "mfsim.nam" ? replace(t, r"BEGIN options"i => "BEGIN options\n  NEWTON", count = 1) :
    endswith(f, ".ims") ? replace(t, r"LINEAR_ACCELERATION\s+CG"i => "LINEAR_ACCELERATION  BICGSTAB") : t
_tv_noint(f, t) = endswith(f, ".sto.tvs") ? replace(t, r"BEGIN options"i => "BEGIN options\n  DISABLE_STORAGE_CHANGE_INTEGRATION", count = 1) : t

# A copy of struct `x` with the fields in `kw` replaced.
_tv_with(x; kw...) = typeof(x)((haskey(kw, f) ? kw[f] : getfield(x, f) for f in fieldnames(typeof(x)))...)

_tv_error(f) = try
    f()
    nothing
catch e
    e
end

# The model of `dir` after `apply_period!` of every period up to `kper` and the start of the
# period's first step (`prepare_time_step`, `prepare_solve`), on the test backend.
function _tv_model(dir, T, kper)
    sim = load_simulation(dir)
    m = GwfModel(TEST_BACKEND, T, only(sim.models), ModflowGPU.nper(sim.tdis))
    steps = time_steps(sim.tdis)
    for i in eachindex(steps)
        s = steps[i]
        s.kper > kper && break
        w = time_step_window(steps, i)
        s.kstp == 1 && apply_period!(m, s.kper; window = w)
        advance_time_series!(m, w)
        start_timestep!(m, s.delt)
        s.kper == kper && break
    end
    return m
end

@testset "TVK6/TVS6: input" begin
    m = only(load_simulation(joinpath(TV_MODELS, "tvk02")).models)
    t = m.npf.tvk
    @test t isa TvSpec && t.kind === :TVK && t.print_input && isempty(t.ts_files)
    @test m.npf.tvk_files == ["gwf_tvk02.npf.tvk"]
    @test t.periods.iper == [2, 3, 4, 5, 7]
    p = t.periods.data[3]
    @test p.nodeu == [1, 3] && p.var == [:K, :K] && p.value == [1.0, 3.0] && p.ts == [Symbol(""), Symbol("")]
    # DISV cellids (layer, cell2d).
    @test only(load_simulation(joinpath(TV_MODELS, "tvk02_disv")).models).npf.tvk.periods.data[3].nodeu == [1, 3]
    t6 = only(load_simulation(joinpath(TV_MODELS, "tvk06ts")).models).npf.tvk
    @test t6.periods.data[1].ts == [:K33_SERIES] && isnan(t6.periods.data[1].value[1])
    @test haskey(t6.timeseries, :K33_SERIES)
    s3 = only(load_simulation(joinpath(TV_MODELS, "tvs03")).models).sto.tvs
    @test s3.kind === :TVS && s3.disable_storage_change_integration && s3.periods.data[3].var == [:SY]
    s2 = only(load_simulation(joinpath(TV_MODELS, "tvs02")).models).sto.tvs
    @test !s2.disable_storage_change_integration && s2.periods.data[1].ts == [:SYH_SERIES]

    mktempdir() do tmp
        dir = _tv_variant(joinpath(TV_MODELS, "tvk02"), joinpath(tmp, "a"))
        tvk = read(joinpath(dir, "gwf_tvk02.npf.tvk"), String)
        w(txt) = write(joinpath(dir, "gwf_tvk02.npf.tvk"), txt)
        w(replace(tvk, "1  1  1  K  3.0" => "1  1  1  KX  3.0"))
        @test occursin("Unrecognized TVK setting 'KX'", input_error_of(() -> load_simulation(dir)).msg)
        w(replace(tvk, "1  1  1  K  3.0" => "1  1  1  K  kser"))
        @test occursin("Expected numeric value for K, but found 'kser'", input_error_of(() -> load_simulation(dir)).msg)
        w(replace(tvk, "1  1  1  K  3.0" => "1  1  4  K  3.0"))
        @test occursin("Cell is outside the model grid: (1,1,4)", input_error_of(() -> load_simulation(dir)).msg)
        w(replace(tvk, "PRINT_INPUT" => "PRINT_INPUT\n  TS6  FILEIN  none.ts"))
        @test input_error_of(() -> load_simulation(dir)) isa MF6InputError
        # Unsupported combinations: XT3D, HFB6, a multi-model solution.
        w(tvk)
        npf = read(joinpath(dir, "gwf_tvk02.npf"), String)
        write(joinpath(dir, "gwf_tvk02.npf"), replace(npf, "BEGIN options" => "BEGIN options\n  XT3D"))
        e = _tv_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 7))
        @test e isa MF6UnsupportedError && occursin("TVK6 with the NPF6 option XT3D", e.msg)
        write(joinpath(dir, "gwf_tvk02.npf"), npf)
        mm = only(load_simulation(dir).models)
        e = _tv_error(() -> solution_setup([mm, mm], ModflowGPU.ExchangeSpec[]))
        @test e isa MF6UnsupportedError && occursin("TVK6 in model", e.msg)
        e = _tv_error(() -> ModflowGPU.tvk_state(_tv_with(mm; hfb = ModflowGPU.HfbSpec(; maxhfb = 1, periods = ModflowGPU.StressPeriods(Int[], ModflowGPU.HfbPeriod[]))), build_connectivity(mm), npf_setup(mm)))
        @test e isa MF6UnsupportedError && occursin("TVK6 with HFB6", e.msg)
        e = _tv_error(() -> ModflowGPU.tvk_state(_tv_with(mm; npf = _tv_with(mm.npf; tvk = nothing)), build_connectivity(mm), npf_setup(mm)))
        @test e isa MF6UnsupportedError && occursin("TVK6 file names without the TVK6 input", e.msg)
    end
    # write_simulation round trip (the TVK6/TVS6 files and their TS6 files).
    for name in ("tvk06ts", "tvs02", "tv_conv", "tvk02_disv")
        a = load_simulation(joinpath(TV_MODELS, name))
        mktempdir() do tmp
            write_simulation(tmp, a)
            @test spec_identical(a, load_simulation(tmp); ignore = (:simdir, :given, :file))
        end
    end
end

@testset "TVK6/TVS6: errors when applied" begin
    mktempdir() do tmp
        # tvk02_negk: K -1 in period 2 (tvk_validate_change).
        dir = _tv_variant(joinpath(TV_MODELS, "tvk02"), joinpath(tmp, "negk"))
        write(joinpath(dir, "gwf_tvk02.npf.tvk"), "BEGIN options\nEND options\n\nBEGIN period  2\n  1  1  1  K  -1.0\n  1  1  3  K33  0.0\nEND period  2\n")
        m = GwfModel(CPU(), Float64, only(load_simulation(dir).models), 7)
        apply_period!(m, 1)
        e = input_error_of(() -> apply_period!(m, 2))
        @test occursin("TVK changed hydraulic property K is <= 0 for cell (1,1,1)    -1.00000", e.msg)
        @test occursin("TVK changed hydraulic property K33 is <= 0 for cell (1,1,3)", e.msg)
        # A cell outside the active domain.
        write(joinpath(dir, "gwf_tvk02.npf.tvk"), "BEGIN options\nEND options\n\nBEGIN period  1\n  1  1  2  K  2.0\nEND period  1\n")
        d = read(joinpath(dir, "gwf_tvk02.dis"), String)
        write(joinpath(dir, "gwf_tvk02.dis"), replace(d, "END griddata" => "  idomain\n    INTERNAL\n      1  0  1\nEND griddata"))
        m = GwfModel(CPU(), Float64, only(load_simulation(dir).models), 7)
        @test occursin("CELLID (1,1,2) is not in the active model domain.", input_error_of(() -> apply_period!(m, 1)).msg)
        # TVS: SY in a model without convertible cells; SS < 0.
        sdir = _tv_variant(joinpath(TV_MODELS, "tvs03"), joinpath(tmp, "sy"))
        sto = read(joinpath(sdir, "gwf_tvs03.sto"), String)
        write(joinpath(sdir, "gwf_tvs03.sto"), replace(sto, "iconvert\n    CONSTANT  1" => "iconvert\n    CONSTANT  0"))
        m = GwfModel(CPU(), Float64, only(load_simulation(sdir).models), 5)
        apply_period!(m, 1)
        e = input_error_of(() -> apply_period!(m, 2))
        @test occursin("TVS cannot change SY for cell (1,1,1) because SY is unused in this model (all ICONVERT flags are 0).", e.msg)
        tvs = read(joinpath(sdir, "gwf_tvs03.sto.tvs"), String)
        write(joinpath(sdir, "gwf_tvs03.sto.tvs"), replace(tvs, "SS  1.0e-5" => "SS  -1.0e-5"))
        write(joinpath(sdir, "gwf_tvs03.sto"), sto)
        m = GwfModel(CPU(), Float64, only(load_simulation(sdir).models), 5)
        foreach(k -> apply_period!(m, k), 1:2)
        @test occursin("TVS changed storage property SS is < 0 for cell (1,1,1)", input_error_of(() -> apply_period!(m, 3)).msg)
    end
end

@testset "TVK6/TVS6: device data" begin
    dir = joinpath(TV_MODELS, "tv_conv")
    spec = only(load_simulation(dir).models)
    conn = build_connectivity(spec)
    nu = conn.nodeuser
    # NPF of the model with the K of period 1 changed by TVK (K of a block of layer-1 cells,
    # a K33 ratio): its saturated conductances are the device's after `apply_period!`. With
    # K33OVERK, K33 is the ratio times K when NPF reads it or TVK sets K33; a later change of
    # K leaves it (tvk_validate_change).
    k = copy(spec.npf.k)
    k33 = spec.npf.k33 .* spec.npf.k
    for (n, v, x) in zip(spec.npf.tvk.periods.data[1].nodeu, spec.npf.tvk.periods.data[1].var, spec.npf.tvk.periods.data[1].value)
        v === :K ? (k[n] = x) : (k33[n] = x * k[n])
    end
    ref = npf_setup(_tv_with(spec; npf = _tv_with(spec.npf; k, k33, k33overk = false)), conn)
    base = npf_setup(spec, conn)
    for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, 4)
        m.offdiag = false
        # The TVK6 file has a TS6 file: the settings apply at every step (`ad`), not at `rp`.
        w = time_step_window(time_steps(load_simulation(dir).tdis), 1)
        apply_period!(m, 1; window = w)
        @test Array(m.npf.condsat) == T.(base.condsat)
        m.offdiag = false
        advance_time_series!(m, w)
        @test isnothing(m.offdiag)
        @test Array(m.npf.condsat) == T.(ref.condsat)
        @test Array(m.npf.cv.condsat) == T.(ref.condsat)
        @test Array(m.npf.cv.hyl) == T.(ref.hyl) && Array(m.npf.cv.hyh) == T.(ref.hyh)
        @test ref.condsat != base.condsat
        # The storage-change data: period 2 changes SS of confined layer-2 cells (and SY):
        # its first step integrates the change, the second does not.
        m = _tv_model(dir, T, 2)
        tv = m.sto.tvs
        @test tv.active
        sc0 = ModflowGPU.storage_capacity(spec.sto, conn)
        n = conn.nodereduced[80 + 2 * 10 + 3]  # (2, 3, 3): SS 4e-4
        rho, rhoold, sdr = Array(m.sto.rho), Array(tv.old.rhoold), Array(tv.old.sdr)
        delt = m.delt
        @test m.sto.sc[n] == 4.0e-4 * (conn.top[n] - conn.bot[n]) * conn.area[n]
        @test rho[n] == T(m.sto.sc[n] * (1.0 / delt))
        @test rhoold[n] == T(sc0[n] * (1.0 / delt))
        @test sdr[n] == T((m.sto.sc[n] * (1.0 / delt) - sc0[n] * (1.0 / delt)) * m.setup.r[n])
        same = [i for i in 1:conn.nodes if m.sto.sc[i] == sc0[i] && (isempty(m.sto.sc2) || m.sto.sc2[i] == spec.sto.sy[nu[i]] * conn.area[i])]
        @test rhoold[same] == rho[same] && all(iszero, sdr[same])
        start_timestep!(m, m.delt * 1.2)
        @test !m.sto.tvs.active
    end
    # Settings equal to the NPF and STO values change nothing: the heads are bitwise those of
    # the model without TVK6 and TVS6.
    mktempdir() do tmp
        none = _tv_variant(dir, joinpath(tmp, "none"); edit = (f, t) -> replace(t, r"\s*TV[KS]6\s+FILEIN\s+\S+" => ""))
        same = _tv_variant(
            dir, joinpath(tmp, "same"); edit = (f, t) -> endswith(f, ".npf.tvk") ?
                "BEGIN options\nEND options\n\nBEGIN period  2\n  1  3  3  K  12.0\n  1  4  4  K22  6.0\n  2  2  2  K33  0.2\nEND period  2\n" :
                endswith(f, ".sto.tvs") ? "BEGIN options\nEND options\n\nBEGIN period  3\n  1  3  3  SS  1.0e-5\n  1  4  4  SY  0.15\nEND period  3\n" : t,
        )
        for_each_precision() do p
            a = ModflowGPU.run(none; backend = TEST_BACKEND, precision = p.preset, write_output = false)
            b = ModflowGPU.run(same; backend = TEST_BACKEND, precision = p.preset, write_output = false)
            @test a.heads == b.heads
        end
    end
end

@testset "TVK6/TVS6: run parity against mf6" begin
    cases = [
        [(c, c, (f, t) -> t) for c in ("tvk01", "tvk02", "tvk02_disv", "tvk03", "tvk04", "tvk05", "tvk06ts")];
        [(c, c, (f, t) -> t) for c in ("tvs01", "tvs02", "tvs03", "tvs04", "tvs05")];
        [("tv_conv", "tv_conv", (f, t) -> t), ("tv_conv_newton", "tv_conv", _tv_newton), ("tv_conv_noint", "tv_conv", _tv_noint)]
    ]
    for (name, src, edit) in cases
        if !test_model_selected(name)
        elseif !isfile(TV_MF6)
            @info "Skipping $name: mf6 not found at $TV_MF6"
        else
            mktempdir() do tmp
                dir = _tv_variant(joinpath(TV_MODELS, src), joinpath(tmp, name); edit)
                out = read(Cmd(`$TV_MF6`; dir), String)
                @test occursin("Normal termination", out)
                @testset "$name" begin
                    # tvs: the one cell's storage flows of period 5 are the difference of two
                    # terms 2e4 times larger (rho2old·thk·snold − rho2·thk·snnew): `:single`
                    # budgets against their rounding bound (T6.1, `_rounding_bounds`).
                    _run_parity(name, dir; presets = TEST_PRESETS, single_rounding_bound = startswith(name, "tvs"))
                end
            end
        end
    end
end
