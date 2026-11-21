# HFB6, horizontal flow barriers (T6.11, src/gwf/hfb.jl):
#
# 1. Input: the HFB6 file into `GwfSpec.hfb` (PERIOD blocks, two cellids per record), MF6's
#    errors (MAXHFB, unconnected cells, two barriers on one connection), a barrier with an
#    excluded cell dropped, XT3D with HFB6 and HFB6 in a multi-model solution unsupported.
# 2. `condsat_modify` (`hfb_condsat`) on the host, and the device conductances after
#    `apply_period!` (static barriers in `condsat`, dynamic ones flagged), periods without a
#    block keeping the barriers and an empty block removing them.
# 3. Run parity against mf6 (run at test time) in `:single` and `:double`:
#    - test/models/hfb_vhfb (MF6 autotest test_gwf_vhfb.py vhfb00: vertical barriers between
#      convertible layers) and its NEWTON (vhfb01) and logarithmic averaging (vhfb04) cases;
#    - test/models/hfb_disu (test_gwf_disu_hfb_idomain.py: DISU, IDOMAIN, user node numbers);
#    - test/models/hfb_conv (convertible DIS, barriers changing between periods: `hfb_fc` and
#      `condsat_modify` barriers, a multiplier, HYDCHR 0) and its NEWTON variant;
#    - ex-gwf-u1disv (reference run inputs) with a line of barriers added (DISV).
using ModflowGPU: load_simulation, build_connectivity, npf_setup, GwfModel, apply_period!, MF6UnsupportedError,
    MF6InputError, hfb_condsat, hfb_barriers, HfbSpec, HfbPeriod, StressPeriods, solution_setup, write_simulation, spec_identical

const HFB_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))
const HFB_MODELS = joinpath(@__DIR__, "..", "models")

# A copy of the simulation `src` (inputs only) in `dst` with `edit(name, text)` applied.
function _hfb_variant(src, dst; edit = (f, t) -> t)
    mkpath(dst)
    for f in readdir(src)
        (f == "generate.jl" || any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".txt", ".toml", ".grb", ".csv"))) && continue
        write(joinpath(dst, f), edit(f, read(joinpath(src, f), String)))
    end
    chmod(dst, 0o755; recursive = true)
    return dst
end

# NEWTON variant: NEWTON in the name file, BICGSTAB.
_hfb_newton(f, t) = endswith(f, ".nam") && f != "mfsim.nam" ? replace(t, r"BEGIN options"i => "BEGIN options\n  NEWTON", count = 1) :
    endswith(f, ".ims") ? replace(t, r"LINEAR_ACCELERATION\s+CG"i => "LINEAR_ACCELERATION  BICGSTAB") : t

_hfb_logarithmic(f, t) = endswith(f, ".npf") ? replace(t, r"BEGIN options"i => "BEGIN options\n  ALTERNATIVE_CELL_AVERAGING  LOGARITHMIC", count = 1) : t

# ex-gwf-u1disv with a line of barriers across the grid: the horizontal connections between
# the cells left and right of the line x = 450 m (positive HYDCHR), and a few connections
# between y-neighbours left of it (negative: a multiplier).
function _u1disv_hfb(dst)
    src = reference_rundir("ex-gwf-u1disv")
    isnothing(src) && return nothing
    _hfb_variant(src, dst)
    m = only(load_simulation(dst).models)
    conn = build_connectivity(m)
    recs = String[]
    ny = 0
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        mm = conn.ja[ii]
        mm > n || continue
        a, b = conn.xc[n], conn.xc[mm]
        if min(a, b) < 450.0 < max(a, b)
            push!(recs, "  1  $(conn.nodeuser[n])  1  $(conn.nodeuser[mm])  2.0e-3")
        elseif ny < 4 && max(a, b) < 300.0 && abs(conn.yc[n] - conn.yc[mm]) > 1.0
            ny += 1
            push!(recs, "  1  $(conn.nodeuser[n])  1  $(conn.nodeuser[mm])  -0.05")
        end
    end
    nam = joinpath(dst, "ex-gwf-u1disv.nam")
    write(nam, replace(read(nam, String), r"END packages"i => "  HFB6  ex-gwf-u1disv.hfb  hfb\nEND packages"))
    write(
        joinpath(dst, "ex-gwf-u1disv.hfb"),
        "BEGIN options\nEND options\n\nBEGIN dimensions\n  MAXHFB  $(length(recs))\nEND dimensions\n\nBEGIN period  1\n$(join(recs, "\n"))\nEND period  1\n",
    )
    return dst
end

_hfb_write(dir, name, txt) = write(joinpath(dir, name), txt)

# The exception `f()` throws (`nothing` when none).
_hfb_error(f) = try
    f()
    nothing
catch e
    e
end

# A model of `dir` at the start of period 1 (HFB6 resolves its barriers then: `hfb_rp`).
_hfb_period1(dir) = apply_period!(GwfModel(CPU(), Float64, only(load_simulation(dir).models), 1), 1)

@testset "HFB6: input" begin
    sim = load_simulation(joinpath(HFB_MODELS, "hfb_conv"))
    m = only(sim.models)
    h = m.hfb
    @test h isa HfbSpec
    @test h.name == "HFB" && h.print_input && h.maxhfb == 23
    @test h.periods.iper == [1, 3, 4]
    p1 = h.periods.data[1]
    @test length(p1.cell1) == 23 && p1.hydchr[1] == 1.0e-3 && p1.hydchr[end] == -0.1
    # (1, 1, 6) and (1, 1, 7) as user nodes; (2, 5, 2) and (2, 6, 2).
    @test (p1.cell1[1], p1.cell2[1]) == (6, 7)
    @test (p1.cell1[end - 2], p1.cell2[end - 2]) == (120 + 4 * 12 + 2, 120 + 5 * 12 + 2)
    @test isempty(h.periods.data[3].cell1)
    d = only(load_simulation(joinpath(HFB_MODELS, "hfb_disu")).models).hfb
    @test d.periods.data[1].cell1 == [26] && d.periods.data[1].cell2 == [27]

    mktempdir() do tmp
        dir = _hfb_variant(joinpath(HFB_MODELS, "hfb_vhfb"), joinpath(tmp, "a"))
        hfb = read(joinpath(dir, "vhfb.hfb"), String)
        # MAXHFB must be positive (`source_dimensions`); more records than MAXHFB.
        _hfb_write(dir, "vhfb.hfb", replace(hfb, "MAXHFB  121" => "MAXHFB  0"))
        e = input_error_of(() -> load_simulation(dir))
        @test e isa MF6InputError && occursin("MAXHFB must be specified with value greater than zero", e.msg)
        _hfb_write(dir, "vhfb.hfb", replace(hfb, "MAXHFB  121" => "MAXHFB  120"))
        @test input_error_of(() -> load_simulation(dir)) isa MF6InputError
        # check_data: unconnected cells; two barriers on one connection.
        _hfb_write(dir, "vhfb.hfb", replace(hfb, "1  1  1  2  1  1  2.0e-5" => "1  1  1  2  1  2  2.0e-5"))
        e = input_error_of(() -> _hfb_period1(dir))
        @test e isa MF6InputError && occursin("HFB no. 1 is between two unconnected cells: (1,1,1) and (2,1,2)", e.msg)
        _hfb_write(dir, "vhfb.hfb", replace(hfb, "1  1  2  2  1  2  2.0e-5" => "2  1  1  1  1  1  2.0e-5"))
        e = input_error_of(() -> _hfb_period1(dir))
        @test e isa MF6InputError && occursin("HFB no. 1 and HFB no. 2 are both between cells (2,1,1) and (1,1,1)", e.msg)
        # XT3D with HFB6 stays unsupported (T6.14).
        _hfb_write(dir, "vhfb.hfb", hfb)
        npf = read(joinpath(dir, "vhfb.npf"), String)
        _hfb_write(dir, "vhfb.npf", replace(replace(npf, "SAVE_FLOWS" => "SAVE_FLOWS\n  XT3D"), "CONSTANT  1\n" => "CONSTANT  0\n"))
        e = _hfb_error(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 1))
        @test e isa MF6UnsupportedError && occursin("HFB6 with the NPF6 option XT3D", e.msg)
        # A barrier with an excluded cell is dropped with a warning (`source_data`).
        ddir = _hfb_variant(joinpath(HFB_MODELS, "hfb_disu"), joinpath(tmp, "d"))
        _hfb_write(ddir, "disuhfb.hfb", replace(read(joinpath(ddir, "disuhfb.hfb"), String), "MAXHFB  1" => "MAXHFB  2", "26  27  1.0e-6" => "26  27  1.0e-6\n  1  2  1.0e-6"))
        dm = only(load_simulation(ddir).models)
        conn = build_connectivity(dm)
        bar = @test_logs (:warn, r"HFB connection between inactive cell\(s\) will be excluded: \(1\) to \(2\)") hfb_barriers(dm.hfb, 1, dm.dis, conn, zeros(Int32, conn.nodes))
        @test bar.noden == [25] && bar.nodem == [26] && bar.static == [true]
    end
    # write_simulation round trip.
    for name in ("hfb_conv", "hfb_disu")
        a = load_simulation(joinpath(HFB_MODELS, name))
        mktempdir() do tmp
            write_simulation(tmp, a)
            @test spec_identical(a, load_simulation(tmp); ignore = (:simdir,))
        end
    end
    # HFB6 in a model of a multi-model solution: unsupported.
    m2 = only(load_simulation(joinpath(HFB_MODELS, "hfb_disu")).models)
    e = _hfb_error(() -> solution_setup([m2, m2], ModflowGPU.ExchangeSpec[]))
    @test e isa MF6UnsupportedError && occursin("HFB6 in model", e.msg)
end

@testset "HFB6: conductances on the device" begin
    sim = load_simulation(joinpath(HFB_MODELS, "hfb_conv"))
    spec = only(sim.models)
    conn = build_connectivity(spec)
    setup = npf_setup(spec, conn)
    ict = setup.properties.icelltype
    for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, 4)
        csat0 = Array(m.npf.cv.condsat)
        apply_period!(m, 1)
        @test isnothing(m.offdiag)
        bar = hfb_barriers(spec.hfb, 1, spec.dis, conn, ict)
        @test count(bar.static) == 9 && count(bar.dynamic) == 14
        cs = Array(m.npf.cv.condsat)
        ihcx = Array(m.npf.cv.ihcx)
        chr = Array(m.npf.cv.hfb)
        @test Array(m.npf.condsat) == cs
        for k in eachindex(bar.jas)
            j = bar.jas[k]
            if bar.static[k]
                ref = hfb_condsat(conn, setup.condsat[j], bar.noden[k], bar.nodem[k], j, bar.hydchr[k])
                @test cs[j] == T(ref)
                @test ihcx[j] & 16 == 0
            else
                @test cs[j] == csat0[j]
                @test ihcx[j] & 16 == 16 && chr[j] == T(bar.hydchr[k])
            end
        end
        others = setdiff(1:conn.njas, bar.jas)
        @test cs[others] == csat0[others] && all(x -> x & 16 == 0, ihcx[others])
        # Period 2 has no block: nothing changes; period 3 replaces the barriers; period 4
        # (an empty block) removes them.
        m.offdiag = false
        apply_period!(m, 2)
        @test m.offdiag === false && Array(m.npf.cv.condsat) == cs
        apply_period!(m, 3)
        bar3 = hfb_barriers(spec.hfb, 2, spec.dis, conn, ict)
        cs3 = Array(m.npf.cv.condsat)
        @test isnothing(m.offdiag)
        for k in eachindex(bar3.jas)
            j = bar3.jas[k]
            bar3.static[k] && @test cs3[j] == (bar3.hydchr[k] == 0 ? zero(T) : T(hfb_condsat(conn, setup.condsat[j], bar3.noden[k], bar3.nodem[k], j, bar3.hydchr[k])))
        end
        @test count(x -> x & 16 != 0, Array(m.npf.cv.ihcx)) == count(bar3.dynamic)
        apply_period!(m, 4)
        @test Array(m.npf.cv.condsat) == csat0 && all(x -> x & 16 == 0, Array(m.npf.cv.ihcx))
    end
    # condsat_modify: series with the barrier over the saturated face area; HYDCHR <= 0 is a multiplier.
    k = findfirst(x -> conn.ihc[x] == 1, 1:conn.njas)
    n = findfirst(n -> any(ii -> conn.jas[ii] == k, (conn.ia[n] + 1):(conn.ia[n + 1] - 1)), 1:conn.nodes)
    ii = findfirst(ii -> conn.jas[ii] == k, (conn.ia[n] + 1):(conn.ia[n + 1] - 1)) + conn.ia[n]
    mm = conn.ja[ii]
    fa = conn.hwva[k] * 0.5 * ((conn.top[n] - conn.bot[n]) + (conn.top[mm] - conn.bot[mm]))
    c = setup.condsat[k]
    @test hfb_condsat(conn, c, n, mm, k, 1.0e-3) ≈ c * (1.0e-3 * fa) / (c + 1.0e-3 * fa) rtol = 1.0e-14
    @test hfb_condsat(conn, c, n, mm, k, -0.25) == 0.25 * c
    @test hfb_condsat(conn, c, n, mm, k, 0.0) == 0.0
end

@testset "HFB6: run parity against mf6" begin
    cases = [
        ("hfb_vhfb", "hfb_vhfb", (f, t) -> t),
        ("hfb_vhfb_newton", "hfb_vhfb", _hfb_newton),
        ("hfb_vhfb_log", "hfb_vhfb", _hfb_logarithmic),
        ("hfb_disu", "hfb_disu", (f, t) -> t),
        ("hfb_conv", "hfb_conv", (f, t) -> t),
        ("hfb_conv_newton", "hfb_conv", _hfb_newton),
        ("hfb_u1disv", nothing, nothing),
    ]
    for (name, src, edit) in cases
        if !test_model_selected(name)
        elseif !isfile(HFB_MF6)
            @info "Skipping $name: mf6 not found at $HFB_MF6"
        else
            mktempdir() do tmp
                dir = isnothing(src) ? _u1disv_hfb(joinpath(tmp, name)) : _hfb_variant(joinpath(HFB_MODELS, src), joinpath(tmp, name); edit)
                if isnothing(dir)
                    @info "Skipping $name: no reference run of ex-gwf-u1disv"
                else
                    out = read(Cmd(`$HFB_MF6`; dir), String)
                    @test occursin("Normal termination", out)
                    @testset "$name" begin
                        _run_parity(name, dir; presets = TEST_PRESETS)
                    end
                end
            end
        end
    end
end
