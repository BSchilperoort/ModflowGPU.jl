# Builder API and MF6 writer (T3.2):
# - test/models/coupling_rde built in memory from Julia arrays (the formulas of its
#   generate.jl, with the 4-digit rounding of the written values) is bitwise identical to
#   the spec `load_simulation` reads from its files (`spec_identical`, every field
#   including `simdir`, file names and record order);
# - spec → `write_simulation` → `load_simulation` round trips (coupling_rde and a model
#   with the other options the writer handles);
# - mf6 6.8.2 runs the written coupling_rde model, and `ModflowGPU.run` of the built spec
#   matches its heads (needs reference/; skips otherwise);
# - validation errors.
using ModflowGPU: build_dis, build_npf, build_ic, build_sto, build_oc, build_boundary, build_array_boundary,
    build_model, build_tdis, build_ims, build_simulation, write_simulation, load_simulation, spec_identical,
    spec_difference, oc_record, open_head_file, read_head, MF6InputError

const BUILDER_RDE_DIR = joinpath(@__DIR__, "..", "models", "coupling_rde")
const BUILDER_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# coupling_rde from Julia arrays (generate.jl; `r` is its 4-digit `fmt`).
function _build_coupling_rde(; simdir = "")
    r(x) = round(x; digits = 4)
    nlay, nrow, ncol = 2, 12, 15
    delr = [100.0 + 10.0 * ((j * 7) % 5) for j in 1:ncol]
    delc = [100.0 + 12.5 * ((i * 3) % 4) for i in 1:nrow]
    top = [22.0 + 0.9 * (j - 1) + 0.15 * (i - 1) + 0.25 * sin(i + 2j) for j in 1:ncol, i in 1:nrow]  # (ncol, nrow)
    idomain = ones(Int, ncol, nrow, nlay)
    for i in 1:nrow, j in 1:ncol
        ((i <= 2 && j >= 13) || (i >= 11 && j <= 2) || (i == 12 && j == 15)) && (idomain[j, i, :] .= 0)
    end
    idomain[9, 6, 1] = 0
    active(k, i, j) = idomain[j, i, k] > 0
    dis = build_dis(; delr, delc, top = r.(top), botm = cat(r.(top .- 10.0), r.(top .- 40.0); dims = 3), idomain, length_units = :meters)
    npf = build_npf(dis; icelltype = 0, k = [12.0, 4.0], k33 = [1.5, 0.5], save_flows = true)
    ic = build_ic(dis; strt = 23.0)
    sto = build_sto(dis; iconvert = 0, ss = 2.0e-5, sy = 0.1, save_flows = true, periods = [1 => :steady_state, 2 => :transient])
    oc = build_oc(; budget_file = "rde.cbc", head_file = "rde.hds", periods = [1 => ["SAVE HEAD ALL", "SAVE BUDGET ALL"]])
    chd_rows = [i for i in 1:nrow if active(1, i, 1)]
    chd = build_boundary(:CHD, dis; name = "chd", periods = [1 => (; cells = [(1, i, 1) for i in chd_rows], values = [r(21.0 + 0.05 * i) for i in chd_rows])])
    wel = build_boundary(
        :WEL, dis; name = "wel", periods = [
            2 => (; cells = [(2, 8, 11), (2, 4, 6)], values = [-900.0, -300.0]),
            3 => (; cells = [(2, 8, 11)], values = [-1500.0]),
        ]
    )
    rcha = build_array_boundary(:RCH, dis; name = "rcha", periods = [1 => (; recharge = 0.0012), 3 => (; recharge = 0.0004)])
    rcells = [(1, i, 6 + (i % 3)) for i in 1:nrow if active(1, i, 6 + (i % 3))]
    stage = [r(top[j, i] - 1.0 - 0.3 * (i % 2)) for (_, i, j) in rcells]
    rvals(st) = hcat(st, [r(150.0 + 10i) for (_, i, _) in rcells], [r(top[j, i] - 2.5) for (_, i, j) in rcells])
    raux = [i % 4 == 0 ? 0.5 : 1.0 for (_, i, _) in rcells]
    riv = build_boundary(
        :RIV, dis; name = "riv", auxnames = ["cmult"], auxmultname = "cmult", boundnames = true, periods = [
            1 => (; cells = rcells, values = rvals(stage), aux = raux, boundnames = fill("river", length(rcells))),
            3 => (; cells = rcells, values = rvals(r.(stage .+ 0.6)), aux = raux, boundnames = fill("river", length(rcells))),
        ]
    )
    dcells = [(1, i, j) for i in 2:2:nrow for j in (3, 4, 5, 10, 11, 12) if active(1, i, j)]
    ddepth = [(i + j) % 3 == 0 ? 0.0 : ((i + j) % 3 == 1 ? 1.5 : -1.0) for (_, i, j) in dcells]
    drn = build_boundary(
        :DRN, dis; name = "drn", auxnames = ["ddrn"], auxdepthname = "ddrn",
        periods = [1 => (; cells = dcells, values = hcat([r(top[j, i] - 0.6) for (_, i, j) in dcells], [r(80.0 + 5j) for (_, _, j) in dcells]), aux = ddepth)]
    )
    ecells = [(1, i, j) for i in 1:2:nrow for j in 1:ncol if active(1, i, j) && !(j in (6, 7, 8))]
    evals(petm0) = [
    hcat([r(top[j, i]) for (_, i, j) in ecells], [r(0.002 + 0.0001 * j) for (_, _, j) in ecells], [i < 7 ? 3.0 : 1.5 for (_, i, _) in ecells])  repeat([0.25 0.6 0.65 0.3 petm0], length(ecells))
    ]
    evt = build_boundary(
        :EVT, dis; name = "evt", nseg = 3, surf_rate_specified = true,
        periods = [1 => (; cells = ecells, values = evals(0.9)), 3 => (; cells = ecells, values = evals(1.0))]
    )
    ievt = ones(Int, ncol, nrow)
    ievt[9, 6] = 2
    evta = build_array_boundary(
        :EVT, dis; name = "evta",
        periods = [1 => (; ievt, surface = r.(top .- 0.3), rate = [iseven(i) ? 0.0015 : 0.0 for j in 1:ncol, i in 1:nrow], depth = 2.0)]
    )
    model = build_model("rde"; dis, npf, ic, sto, oc, boundaries = [chd, wel, rcha, riv, drn, evt, evta], save_flows = true)
    tdis = build_tdis(; perlen = [1.0, 30.0, 20.0], nstp = [1, 5, 4], tsmult = [1.0, 1.2, 1.0], time_units = :days)
    ims = build_ims(;
        print_option = :summary, outer_dvclose = 1.0e-10, outer_maximum = 200, inner_maximum = 1000,
        inner_dvclose = 1.0e-13, inner_rclose = 1.0e-10, linear_acceleration = :bicgstab,
    )
    return build_simulation(model; tdis, ims, simdir)
end

# A model with the other options the writer handles (loaded with strict = false where the
# engine does not support them).
function _build_options_model()
    ncol, nrow, nlay = 6, 5, 3
    top = [10.0 + 0.1j - 0.05i for j in 1:ncol, i in 1:nrow]
    botm = cat(top .- 3.0, fill(2.0, ncol, nrow), fill(-5.0, ncol, nrow); dims = 3)
    mask = trues(ncol, nrow)
    mask[1, 1] = false
    dis = build_dis(; delr = 50.0, delc = [40.0, 41.0, 42.0, 43.0, 44.0], top, botm, idomain = Int.(mask), xorigin = 100.0, angrot = 12.5, nogrb = true)
    npf = build_npf(dis; k = reshape(collect(1.0:90.0), ncol, nrow, nlay), k22 = [1.0, 2.0, 3.0], cell_averaging = :logarithmic, save_specific_discharge = true)
    ic = build_ic(dis; strt = [9.0, 8.0, 7.0])
    sto = build_sto(dis; ss = 1.0e-5, storagecoefficient = true, periods = [2 => :transient])
    oc = build_oc(; head_file = "o.hds", periods = [1 => [oc_record(:save, "HEAD", :frequency, [2]), "PRINT BUDGET STEPS 1 3"], 3 => String[]])
    cells = [CartesianIndex(2, 3), CartesianIndex(4, 5)]
    ghb = build_boundary(:GHB, dis; periods = [1 => (; cells, layer = [1, 3], values = [9.5 100.0; 8.0 1.0e-3])], print_input = true)
    rch = build_boundary(:RCH, dis; periods = [1 => (; cells = [7, 8, 9], values = [1.0e-3, 2.0e-3, 3.0e-3])])
    w1 = build_boundary(:WEL, dis; periods = [2 => (; cells = [CartesianIndex(3, 3, 3)], values = [-10.0])])
    w2 = build_boundary(:WEL, dis; maxbound = 5, auxnames = ["a", "b"], periods = [1 => (; cells = [(2, 2, 2)], values = [-1.0], aux = [1.0 2.0]), 3 => (; cells = Int[], values = zeros(0, 1), aux = zeros(0, 2))])
    chd = build_boundary(:CHD, dis; save_flows = true, periods = [1 => (; cells = [(1, 5, 6)], values = [9.0])])
    rcha = build_array_boundary(:RCH, dis; name = "rcha", auxnames = ["m"], auxmultname = "m", periods = [1 => (; irch = 2, recharge = top ./ 1.0e4, m = 0.5)])
    model = build_model("opts"; dis, npf, ic, sto, oc, boundaries = [chd, w1, ghb, w2, rch, rcha], print_flows = true, list_file = "opts.lst")
    tdis = build_tdis(; perlen = [1.0, 10.0, 5.0], nstp = [1, 4, 2], tsmult = [1.0, 1.5, 1.0], start_date_time = "2020-01-01")
    ims = build_ims(; complexity = :moderate, inner_rclose = 1.0e-6, rclose_option = :strict, scaling_method = :diagonal, csv_outer_output = "outer.csv")
    return build_simulation(model; tdis, ims, continue_on_nonconvergence = true, maxerrors = 7)
end

@testset "builder" begin
    @testset "coupling_rde in memory == from files (bitwise)" begin
        loaded = load_simulation(BUILDER_RDE_DIR)
        built = _build_coupling_rde(; simdir = loaded.simdir)
        d = spec_difference(built, loaded)
        isnothing(d) || @info "coupling_rde: first difference: $d"
        @test isnothing(d)
        m = only(built.models)
        @test [b.name for b in m.boundaries] == ["WEL", "DRN", "RIV", "RCHA", "EVT", "EVTA", "CHD"]
        @test [e.name for e in m.packages] == ["DIS", "NPF", "IC", "STO", "CHD", "WEL", "RCHA", "RIV", "DRN", "EVT", "EVTA", "OC"]
    end

    @testset "spec_difference" begin
        a = _build_coupling_rde()
        @test spec_identical(a, _build_coupling_rde())
        @test !spec_identical(a, _build_coupling_rde(; simdir = "x"))
        @test spec_identical(a, _build_coupling_rde(; simdir = "x"); ignore = (:simdir,))
        b = _build_coupling_rde()
        b.models[1].npf.k[3] = prevfloat(b.models[1].npf.k[3])
        @test occursin("spec.models[1].npf.k[3]", spec_difference(a, b))
        c = _build_coupling_rde()
        c.models[1].ic.strt[1] = -0.0
        c.models[1].ic.strt[2] = 0.0
        a.models[1].ic.strt[1:2] .= 0.0
        @test occursin("strt[1]", spec_difference(a, c))  # -0.0 vs 0.0
        @test isnothing(spec_difference(NaN, NaN)) && !isnothing(spec_difference(Int32[1], [1]))
    end

    @testset "write → load round trip" begin
        mktempdir() do dir
            for (name, spec, strict) in (("coupling_rde", _build_coupling_rde(), true), ("options", _build_options_model(), false))
                sub = joinpath(dir, name)
                write_simulation(sub, spec)
                back = load_simulation(sub; strict)
                d = spec_difference(spec, back; ignore = (:simdir,))
                isnothing(d) || @info "$name round trip: first difference: $d"
                @test isnothing(d)
                # Writing the loaded spec gives the same files.
                sub2 = joinpath(dir, name * "2")
                write_simulation(sub2, back)
                @test all(f -> read(joinpath(sub, f)) == read(joinpath(sub2, f)), readdir(sub))
            end
        end
    end

    @testset "mf6 runs the written model; ModflowGPU.run matches" begin
        if !isfile(BUILDER_MF6)
            @info "Skipping the builder mf6 run: mf6 not found at $BUILDER_MF6 (set REFERENCE_DIR or MF6)"
        else
            mktempdir() do dir
                spec = _build_coupling_rde()
                write_simulation(dir, spec)
                cmd = Cmd(`$BUILDER_MF6`; dir)
                out = read(pipeline(cmd; stderr = devnull), String)
                @test occursin("Normal termination", out)
                hds = open_head_file(joinpath(dir, "rde.hds"))
                h6 = vec(read_head(hds, length(hds)))
                act = h6 .!= 1.0e30  # heads of the active cells in user order = reduced nodes
                for_each_precision() do p
                    r = ModflowGPU.run(spec; backend = TEST_BACKEND, precision = p.preset, write_output = false)
                    dh = maximum(abs.(r.heads .- h6[act]))
                    @test dh < (p.T === Float64 ? 1.0e-6 : 1.0e-3)
                    @info "builder: coupling_rde built in memory vs mf6 6.8.2 on the written files, :$(p.preset): max |Δh| = $(round(dh; sigdigits = 3)) m"
                end
            end
            mktempdir() do dir  # mf6 accepts the other options the writer writes
                write_simulation(dir, _build_options_model())
                out = read(pipeline(Cmd(`$BUILDER_MF6`; dir); stderr = devnull), String)
                @test occursin("Normal termination", out)
            end
        end
    end

    @testset "validation" begin
        botm = fill(0.0, 3, 2, 1)
        @test_throws ArgumentError build_dis(; delr = 1.0, delc = [1.0], top = 1.0, botm)  # delc length
        @test_throws ArgumentError build_dis(; delr = -1.0, delc = 1.0, top = 1.0, botm)
        @test_throws ArgumentError build_dis(; delr = 1.0, delc = 1.0, top = NaN, botm)
        @test_throws MF6InputError build_dis(; delr = 1.0, delc = 1.0, top = -1.0, botm)  # thickness
        dis = build_dis(; delr = 1.0, delc = 1.0, top = 1.0, botm, idomain = [1 1; 1 1; 0 1])
        @test_throws ArgumentError build_npf(dis; k = [1.0, 2.0])  # 2 values, 1 layer, 6 cells
        @test_throws ArgumentError build_npf(dis; k = 0.0)
        @test_throws ArgumentError build_npf(dis; k = 1.0, bogus = true)
        @test_throws ArgumentError build_npf(dis; k = 1.0, icelltype = 0.5)
        @test_throws ArgumentError build_sto(dis; ss = 1.0, periods = [1 => :steady])
        @test_throws ArgumentError build_sto(dis; ss = 1.0, iconvert = 1)  # no sy
        @test_throws ArgumentError build_sto(dis; ss = 1.0, periods = [1 => :transient, 1 => :transient])
        @test_throws ArgumentError oc_record(:save, "FLOW")
        @test_throws ArgumentError build_boundary(:XYZ, dis; periods = [])
        @test_throws ArgumentError build_boundary(:CHD, dis; periods = [1 => (; cells = [(1, 1, 3)], values = [1.0])])  # inactive
        @test_throws ArgumentError build_boundary(:CHD, dis; periods = [1 => (; cells = [(1, 3, 1)], values = [1.0])])  # outside
        @test_throws ArgumentError build_boundary(:RIV, dis; periods = [1 => (; cells = [(1, 1, 1)], values = [1.0 2.0])])  # 3 columns
        @test_throws ArgumentError build_boundary(:WEL, dis; maxbound = 1, periods = [1 => (; cells = [1, 2], values = [1.0, 2.0])])
        @test_throws ArgumentError build_boundary(:WEL, dis; auxnames = ["a"], periods = [1 => (; cells = [1], values = [1.0])])  # no aux
        @test_throws ArgumentError build_boundary(:WEL, dis; auxmultname = "m", periods = [])
        @test_throws ArgumentError build_boundary(:RIV, dis; boundnames = true, periods = [1 => (; cells = [1], values = [1.0 1.0 0.0], boundnames = ["a b"])])
        @test_throws ArgumentError build_array_boundary(:WEL, dis; periods = [])
        @test_throws ArgumentError build_array_boundary(:RCH, dis; periods = [1 => (; irch = 2)])  # layer 2 of 1
        @test_throws ArgumentError build_array_boundary(:RCH, dis; periods = [1 => (; rate = 1.0)])
        @test_throws MF6InputError build_tdis(; perlen = [1.0], nstp = 0)
        @test_throws ArgumentError build_ims(; outer_maximum = 0)
        @test_throws ArgumentError build_ims(; linear_acceleration = :gmres)
        @test_throws ArgumentError build_ims(; models = ["x"], given = String[])
        npf, ic = build_npf(dis; k = 1.0), build_ic(dis; strt = 0.5)
        chd = build_boundary(:CHD, dis; periods = [1 => (; cells = [1], values = [0.5])])
        @test_throws ArgumentError build_model("a b"; dis, npf, ic)
        @test [b.name for b in build_model("m"; dis, npf, ic, boundaries = [chd, chd]).boundaries] == ["CHD-1", "CHD-2"]
        chdc = build_boundary(:CHD, dis; name = "c", periods = [1 => (; cells = [1], values = [0.5])])
        @test_throws ArgumentError build_model("m"; dis, npf, ic, boundaries = [chdc, chdc])
        m = build_model("m"; dis, npf, ic, boundaries = [build_boundary(:CHD, dis; periods = [4 => (; cells = [1], values = [0.5])])])
        @test_throws ArgumentError build_simulation(m; tdis = build_tdis(; perlen = [1.0, 1.0]))  # PERIOD 4 > NPER
        dis2 = build_dis(; delr = 1.0, delc = 1.0, top = 1.0, botm = fill(0.0, 2, 2, 1))
        @test_throws ArgumentError build_model("m"; dis = dis2, npf, ic)  # 6 vs 4 cells
    end
end
