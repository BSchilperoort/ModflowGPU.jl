# Tests of the MF6 output files (T1.5): head, budget and binary grid files, and the
# budget tables of the listing file. Synthetic round trips always run; the checks on the
# reference runs of T0.3 (reference/runs) skip with a message when they are missing.

using TOML: TOML

const MO = ModflowGPU

# A minimal labeled budget table as `budget_ot` (Budget.f90) prints it, with F17.4 and
# 1PE17.4 values and two terms of the same name.
function _listing_table(kstp, kper; zone = "ENTIRE MODEL")
    row(name, cum, rate, pkg) = " " * "   " * lpad(name, 16) * " =" * lpad(cum, 17) * " "^6 * lpad(name, 16) * " =" * lpad(rate, 17) * " "^5 * rpad(pkg, 33)
    return join(
        [
            "",
            "",
            "  VOLUME BUDGET FOR $zone AT END OF TIME STEP$(lpad(kstp, 5)), STRESS PERIOD$(lpad(kper, 4))",
            "  " * "-"^99,
            "",
            "     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T          PACKAGE NAME    ",
            "     ------------------                 ------------------------                     ----------------",
            "",
            "           IN:                                      IN:",
            "           ---                                      ---",
            row("STO-SS", "1.2500E-02", "2.5000E-03", "STORAGE"),
            row("RCH", "100.0000", "10.0000", "RCH-1"),
            row("RCH", "50.0000", "5.0000", "RCH-2"),
            row("CHD", "0.0000", "0.0000", "CHD_0"),
            "",
            "            TOTAL IN =         150.0125              TOTAL IN =          15.0025",
            "",
            "          OUT:                                     OUT:",
            "          ----                                     ----",
            row("STO-SS", "0.0000", "0.0000", "STORAGE"),
            row("RCH", "0.0000", "0.0000", "RCH-1"),
            row("RCH", "0.0000", "0.0000", "RCH-2"),
            row("CHD", "150.0000", "15.0000", "CHD_0"),
            "",
            "           TOTAL OUT =         150.0000             TOTAL OUT =          15.0000",
            "",
            "            IN - OUT =       1.2500E-02              IN - OUT =       2.5000E-03",
            "",
            " PERCENT DISCREPANCY =           0.01     PERCENT DISCREPANCY =           0.02",
            "",
        ], "\n",
    )
end

@testset "output: text fields" begin
    @test MO.pad_text("HEAD") == "HEAD            "
    @test MO.pad_text("STO-SS"; justify = :right) == "          STO-SS"
    @test MO.pad_text("    FLOW-JA-FACE"; justify = :left) == "    FLOW-JA-FACE"
    @test_throws ArgumentError MO.pad_text("A"^17)
    @test_throws ArgumentError MO.pad_text("x"; justify = :center)
end

@testset "output: head file round trip" begin
    dir = mktempdir()
    # (grid, values shape, expected record layout (m1, m2) and layer count)
    layouts = [
        (:dis, (4, 3, 2), (4, 3), 2),
        (:disv, (5, 3), (5, 1), 3),
        (:disu, (7,), (7, 1), 1),
    ]
    for_each_precision() do p
        for (grid, shape, (m1, m2), nlay) in layouts
            @testset "$grid" begin
                path = joinpath(dir, "heads_$(grid)_$(p.preset).hds")
                times = [(1, 1, 1.0, 1.0), (1, 2, 0.5, 1.5), (2, 2, 1.25, 2.25)]
                arrays = [p.T.(reshape(range(-3.5, 9.25; length = prod(shape)), shape) .+ k) for k in 1:3]
                # Inactive and dry cells as MF6 writes them (written as Float64 constants).
                values = [Float64.(a) for a in arrays]
                values[1][1] = MO.HNOFLO
                values[2][end] = MO.HDRY
                open(path, "w") do io
                    for (k, (kstp, kper, pertim, totim)) in enumerate(times)
                        MO.write_head_record(io, k == 3 ? arrays[k] : values[k]; kstp, kper, pertim, totim)
                    end
                end
                @test filesize(path) == 3 * nlay * (MO.HEAD_HEADER_BYTES + 8 * m1 * m2)
                hf = MO.open_head_file(path; grid)
                @test length(hf) == 3
                @test MO.head_kstpkper(hf) == [(1, 1), (1, 2), (2, 2)]
                @test MO.head_times(hf) == [1.0, 1.5, 2.25]
                for k in 1:3
                    h = MO.read_head(hf, k)
                    @test h isa Array{Float64}
                    @test size(h) == shape
                    @test h == (k == 3 ? Float64.(arrays[k]) : values[k])
                    s = hf.steps[k]
                    @test (s.m1, s.m2, s.nlay, s.text) == (m1, m2, nlay, "HEAD            ")
                    @test s.pertim == times[k][3]
                end
                @test MO.read_head(hf; kstp = 1, kper = 2) == values[2]
                @test MO.read_head(hf; totim = 2.25, text = "head") == Float64.(arrays[3])
                @test_throws ArgumentError MO.read_head(hf; kper = 3)
                @test_throws ArgumentError MO.read_head(hf; kstp = 1)
                # Raw layout: header of the last layer record of the first array.
                open(path) do io
                    seek(io, (nlay - 1) * (MO.HEAD_HEADER_BYTES + 8 * m1 * m2))
                    @test read(io, Int32) == 1 && read(io, Int32) == 1
                    @test read(io, Float64) == 1.0 && read(io, Float64) == 1.0
                    @test String(read(io, 16)) == "HEAD            "
                    @test (read(io, Int32), read(io, Int32), read(io, Int32)) == (m1, m2, nlay)
                end
                # The :auto layout is the record layout (m1, m2, nlay).
                @test size(MO.read_head(MO.open_head_file(path), 1)) == (m1, m2, nlay)
                # Rewriting through the reader gives the same bytes.
                copy = MO.write_head_file(joinpath(dir, "copy.hds"), hf)
                @test read(copy) == read(path)
            end
        end
    end
    # Malformed files
    path = joinpath(dir, "bad.hds")
    open(io -> MO.write_head_record(io, zeros(3, 2, 2); kstp = 1, kper = 1, pertim = 1.0, totim = 1.0), path, "w")
    bytes = read(path)
    write(path, bytes[1:(end - 8)])
    @test_throws MO.MF6BinaryFileError MO.open_head_file(path)
    write(path, bytes[(MO.HEAD_HEADER_BYTES + 48 + 1):end])  # starts with layer 2
    @test_throws MO.MF6BinaryFileError MO.open_head_file(path)
    @test_throws ArgumentError MO.open_head_file(path; grid = :foo)
end

@testset "output: explicit layouts and sentinels" begin
    dir = mktempdir()
    # Layouts from specs and arrays
    dis = MO.DisSpec(; nlay = 2, nrow = 3, ncol = 4, delr = ones(4), delc = ones(3), top = zeros(4, 3), botm = fill(-1.0, 4, 3, 2))
    L = MO.OutputLayout(dis)
    @test L == MO.OutputLayout(:dis, 4, 3, 2)
    @test MO.nodesuser(L) == 24 && MO.budget_dims(L) == (4, 3, 2)
    @test MO.OutputLayout(zeros(5, 3)) == MO.OutputLayout(:disv, 5, 1, 3)
    @test MO.OutputLayout(zeros(7)) == MO.OutputLayout(:disu, 7, 1, 1)
    @test_throws ArgumentError MO.OutputLayout(:disv, 5, 2, 3)
    @test_throws ArgumentError MO.OutputLayout(:disu, 5, 1, 2)
    @test_throws ArgumentError MO.OutputLayout(:foo, 5, 1, 1)
    kw = (; kstp = 1, kper = 1, pertim = 1.0, totim = 1.0)
    for_each_precision() do p
        h = p.T.(reshape(1:24, 4, 3, 2)) ./ p.T(8)
        # A flat user-node vector with an explicit layout gives the same bytes as the 3-D array.
        a, b = joinpath(dir, "a.hds"), joinpath(dir, "b.hds")
        open(io -> MO.write_head_record(io, L, vec(h); kw...), a, "w")
        open(io -> MO.write_head_record(io, h; kw...), b, "w")
        @test read(a) == read(b)
        @test_throws DimensionMismatch MO.write_head_record(IOBuffer(), L, vec(h)[1:23]; kw...)
        @test_throws DimensionMismatch MO.write_head_record(IOBuffer(), MO.OutputLayout(:disv, 12, 1, 3), h; kw...)
        # Sentinels from masks are exact Float64 HNOFLO/HDRY, also for Float32 heads.
        inactive = falses(24)
        inactive[[1, 13]] .= true
        dry = falses(24)
        dry[5] = true
        open(io -> MO.write_head_record(io, L, h; kw..., inactive, dry), a, "w")
        r = vec(MO.read_head(MO.open_head_file(a), 1))
        @test r[[1, 13]] == [MO.HNOFLO, MO.HNOFLO]
        @test r[5] == MO.HDRY
        keep = setdiff(1:24, [1, 5, 13])
        @test r[keep] == Float64.(vec(h)[keep])
        # Reduced values with nodeuser: missing user nodes are HNOFLO, dry by reduced index.
        nodeuser = [2, 3, 7, 24]
        open(io -> MO.write_head_record(io, L, p.T[1.5, 2.5, 3.5, 4.5]; kw..., nodeuser, dry = [false, true, false, false]), a, "w")
        r = vec(MO.read_head(MO.open_head_file(a), 1))
        @test r[nodeuser] == [1.5, MO.HDRY, 3.5, 4.5]
        @test all(==(MO.HNOFLO), r[setdiff(1:24, nodeuser)])
        @test_throws ArgumentError MO.head_output_values(L, [1.0]; nodeuser = [25])
        @test_throws DimensionMismatch MO.head_output_values(L, [1.0, 2.0]; nodeuser = [1])
    end
    # Why the masks: a Float32 sentinel is not HNOFLO after conversion to Float64.
    @test Float64(Float32(MO.HNOFLO)) != MO.HNOFLO
    # Budget arrays and lists with a layout; names are upper-cased as in MF6.
    a, b = joinpath(dir, "a.cbc"), joinpath(dir, "b.cbc")
    bkw = (; kstp = 1, kper = 1, delt = 1.0, pertim = 1.0, totim = 1.0)
    sto = reshape(collect(1.0:24.0), 4, 3, 2)
    l = MO.BudgetList(Int32[3], Int32[1], [-2.0])
    open(a, "w") do io
        MO.write_budget_array(io, L, "STO-SS", vec(sto); bkw...)
        MO.write_budget_list(io, L, "WEL", l; bkw..., modelnam1 = "model", paknam1 = "model", modelnam2 = "model", paknam2 = "wel-1")
    end
    open(b, "w") do io
        MO.write_budget_array(io, "STO-SS", sto; bkw...)
        MO.write_budget_list(io, "WEL", l; bkw..., dims = (4, 3, 2), modelnam1 = "MODEL", paknam1 = "MODEL", modelnam2 = "MODEL", paknam2 = "WEL-1")
    end
    @test read(a) == read(b)
    r = MO.open_budget_file(a).records[2]
    @test (r.modelnam1, r.paknam2, r.ndim1, r.ndim2, r.ndim3) == ("MODEL           ", "WEL-1           ", 4, 3, -2)
end

@testset "output: budget file round trip" begin
    dir = mktempdir()
    # (grid, grid dims, a STO array of the grid shape, nja)
    layouts = [
        (:dis, (4, 3, 2), 70),
        (:disv, (5, 1, 3), 41),
        (:disu, (7, 1, 1), 19),
    ]
    for_each_precision() do p
        for (grid, dims, nja) in layouts
            @testset "$grid" begin
                path = joinpath(dir, "budget_$(grid)_$(p.preset).cbc")
                nodes = prod(dims)
                sto = grid === :dis ? zeros(p.T, dims) : grid === :disv ? zeros(p.T, dims[1], dims[3]) : zeros(p.T, nodes)
                sto .= p.T.(reshape(range(-1, 1; length = nodes), size(sto)))
                flowja = p.T.(collect(range(-2.5, 3.5; length = nja)))
                wel = MO.BudgetList(Int32[2, 5], Int32[1, 2], [-100.0, -0.125])
                rch = MO.BudgetList(Int32[1, 2, 3], Int32[1, 2, 3], [0.5, 0.25, 0.0], [1.0 2.0 3.0; -1.0 -2.0 -3.0])
                exg = MO.BudgetList(Int32[nodes], Int32[1], [1.0e-3], [45.0 0.5]')
                base = (; delt = 1.0, pertim = 1.0, totim = 1.0)
                open(path, "w") do io
                    for (kstp, kper, totim) in ((1, 1, 1.0), (1, 2, 3.0))
                        kw = (; base..., kstp, kper, totim)
                        MO.write_budget_array(io, "FLOW-JA-FACE", flowja; kw...)
                        MO.write_budget_array(io, "STO-SS", sto; kw...)
                        MO.write_budget_list(io, "WEL", wel; kw..., dims, modelnam1 = "MODEL", paknam1 = "MODEL", modelnam2 = "MODEL", paknam2 = "WEL_0")
                        MO.write_budget_list(io, "RCH", rch; kw..., dims, modelnam1 = "MODEL", paknam1 = "MODEL", modelnam2 = "MODEL", paknam2 = "RCH-1", auxtxt = ["MULTIPLIER", "             AUX"])
                        MO.write_budget_list(io, "FLOW-JA-FACE", exg; kw..., dims, modelnam1 = "MODEL", paknam1 = "GWF-GWF_1", modelnam2 = "OTHER", paknam2 = "GWF-GWF_1", auxtxt = ["ANGLDEGX", "CDIST"])
                    end
                end
                bf = MO.open_budget_file(path)
                @test length(bf) == 10
                @test MO.budget_texts(bf) == ["FLOW-JA-FACE", "STO-SS", "WEL", "RCH"]
                @test MO.budget_kstpkper(bf) == [(1, 1), (1, 2)]
                @test MO.budget_times(bf) == [1.0, 3.0]
                r = bf.records[1]
                @test (r.text, r.ndim1, r.ndim2, r.ndim3, r.imeth, r.n) == ("    FLOW-JA-FACE", nja, 1, -1, 1, nja)
                r = bf.records[2]
                @test (r.text, r.ndim1, r.ndim2, r.ndim3, r.imeth, r.n) == ("          STO-SS", dims[1], dims[2], -dims[3], 1, nodes)
                r = bf.records[4]
                @test (r.text, r.ndim1, r.ndim2, r.ndim3, r.imeth, r.n) == ("             RCH", dims[1], dims[2], -dims[3], 6, 3)
                @test (r.modelnam1, r.paknam1, r.modelnam2, r.paknam2) == ("MODEL           ", "MODEL           ", "MODEL           ", "RCH-1           ")
                @test r.auxtxt == ["MULTIPLIER      ", "             AUX"]
                @test (r.delt, r.pertim, r.totim) == (1.0, 1.0, 1.0)
                @test MO.read_budget(bf, 1) == Float64.(flowja)
                @test reshape(MO.read_budget(bf, 2), size(sto)) == Float64.(sto)
                @test MO.read_budget(bf, 3) == wel
                @test MO.read_budget(bf, 4) == rch
                @test MO.read_budget(bf, 5) == exg
                @test MO.find_budget_records(bf; text = "flow-ja-face") == [1, 5, 6, 10]
                @test MO.find_budget_records(bf; text = "FLOW-JA-FACE", imeth = 1, kper = 2) == [6]
                @test MO.find_budget_records(bf; paknam2 = "GWF-GWF_1", modelnam2 = "OTHER") == [5, 10]
                @test MO.read_budget(bf; text = "WEL", totim = 3.0) == [wel]
                @test isempty(MO.read_budget(bf; text = "DRN"))
                # Rewriting through the reader gives the same bytes.
                copy = MO.write_budget_file(joinpath(dir, "copy.cbc"), bf)
                @test read(copy) == read(path)
            end
        end
    end
    # Argument checks and malformed files
    io = IOBuffer()
    @test_throws DimensionMismatch MO.write_budget_array(io, "STO-SS", zeros(4); kstp = 1, kper = 1, delt = 1, pertim = 1, totim = 1, dims = (3, 1, 1))
    @test_throws DimensionMismatch MO.write_budget_array(io, MO.OutputLayout(:dis, 2, 2, 2), "STO-SS", zeros(4); kstp = 1, kper = 1, delt = 1, pertim = 1, totim = 1)
    l = MO.BudgetList(Int32[1], Int32[1], [1.0], zeros(2, 1))
    @test_throws ArgumentError MO.write_budget_list(io, "WEL", l; kstp = 1, kper = 1, delt = 1, pertim = 1, totim = 1, dims = (1, 1, 1), modelnam1 = "M", paknam1 = "M", modelnam2 = "M", paknam2 = "W", auxtxt = ["A"])
    path = joinpath(dir, "bad.cbc")
    open(io -> MO.write_budget_list(io, "WEL", l; kstp = 1, kper = 1, delt = 1, pertim = 1, totim = 1, dims = (1, 1, 1), modelnam1 = "M", paknam1 = "M", modelnam2 = "M", paknam2 = "W", auxtxt = ["A", "B"]), path, "w")
    write(path, read(path)[1:(end - 1)])
    @test_throws MO.MF6BinaryFileError MO.open_budget_file(path)
    open(io -> write(io, Int32(1), Int32(1), "             WEL", Int32(1), Int32(1), Int32(1)), path, "w")
    @test_throws MO.MF6BinaryFileError MO.open_budget_file(path)  # full (non-compact) header
end

@testset "output: binary grid file round trip" begin
    hdr(s) = rpad(s, 49) * "\n"
    def(s) = rpad(s, 99) * "\n"
    vars = [
        MO.GrbVariable("NODES", :INTEGER, Int[], def("NODES INTEGER NDIM 0 # 3")),
        MO.GrbVariable("TOP", :DOUBLE, [3], def("TOP DOUBLE NDIM 1 3")),
        MO.GrbVariable("VERTICES", :DOUBLE, [2, 2], def("VERTICES DOUBLE NDIM 2 2 2")),
    ]
    data = Dict{String, Any}("NODES" => Int32(3), "TOP" => [1.0, 2.0, 3.5], "VERTICES" => [0.0 1.0; 2.0 3.0])
    g = MO.BinaryGridFile("DISU", 1, [hdr("GRID DISU"), hdr("VERSION 1"), hdr("NTXT 3"), hdr("LENTXT 100")], 100, vars, data)
    path = MO.write_grb(joinpath(mktempdir(), "x.disu.grb"), g)
    @test filesize(path) == 4 * 50 + 3 * 100 + 4 + 3 * 8 + 4 * 8
    h = MO.read_grb(path)
    @test (h.grid, h.version, h.lentxt) == ("DISU", 1, 100)
    @test collect(keys(h)) == ["NODES", "TOP", "VERTICES"]
    @test h["NODES"] === Int32(3)
    @test h["TOP"] == [1.0, 2.0, 3.5]
    @test h["VERTICES"] == [0.0 1.0; 2.0 3.0]
    @test haskey(h, "TOP") && !haskey(h, "BOTM")
    write(path, read(path)[1:(end - 1)])
    @test_throws MO.MF6BinaryFileError MO.read_grb(path)
end

@testset "output: listing budget tables" begin
    path = joinpath(mktempdir(), "model.lst")
    write(path, "header text\n" * _listing_table(1, 1) * "\n other output\n" * _listing_table(1, 1; zone = "ZONE 7") * _listing_table(3, 12))
    tables = MO.read_listing_budgets(path)
    @test length(tables) == 3
    @test tables[2].zone == "ZONE 7"
    b = tables[3]
    @test (b.title, b.zone, b.kstp, b.kper) == ("VOLUME", "ENTIRE MODEL", 3, 12)
    @test [t.name for t in b.terms] == ["STO-SS", "RCH", "RCH", "CHD"]
    @test [t.package for t in b.terms] == ["STORAGE", "RCH-1", "RCH-2", "CHD_0"]
    @test b.terms[1].cum_in == 1.25e-2 && b.terms[1].rate_in == 2.5e-3
    @test b.terms[4].cum_out == 150.0 && b.terms[4].rate_out == 15.0
    @test (b.total_cum_in, b.total_cum_out, b.total_rate_in, b.total_rate_out) == (150.0125, 150.0, 15.0025, 15.0)
    @test (b.cum_in_minus_out, b.rate_in_minus_out) == (1.25e-2, 2.5e-3)
    @test (b.cum_percent_discrepancy, b.rate_percent_discrepancy) == (0.01, 0.02)
    @test MO.listing_rates(path, "RCH") == [(15.0, 0.0), (15.0, 0.0)]
    @test MO.listing_rates(path, "rch-2") == [(5.0, 0.0), (5.0, 0.0)]
    @test MO.listing_rates(path, "CHD") == [(0.0, 15.0), (0.0, 15.0)]
    @test MO.listing_rates(path, "CHD"; zone = "zone 7") == [(0.0, 15.0)]
    @test isempty(MO.listing_rates(path, "CHD"; title = "MASS"))
    @test_throws ArgumentError MO.listing_rates(path, "WEL")
    @test MO.percent_discrepancy(15.0025, 15.0) ≈ 100 * 0.0025 / 15.00125
    @test MO.percent_discrepancy(0.0, 0.0) == 0.0
    write(path, replace(_listing_table(1, 1), "TOTAL OUT" => "TOTAL XXX"))
    @test_throws MO.MF6InputError MO.read_listing_budgets(path)
end

# Opt-in (MFGPU_FLOPY=1, needs `uv`; never in CI): flopy reads the files our writers produce.
@testset "output: files readable by flopy" begin
    if get(ENV, "MFGPU_FLOPY", "0") != "1"
        @info "Skipping the flopy check: set MFGPU_FLOPY=1 to run it (needs uv; downloads flopy and numpy)"
    elseif Sys.which("uv") === nothing
        @info "Skipping the flopy check: uv not found on PATH"
    else
        dir = mktempdir()
        L = MO.OutputLayout(:dis, 4, 3, 2)
        heads = [collect(1.0:24.0) .+ k / 4 for k in 1:2]
        inactive = falses(24)
        inactive[1] = true
        open(joinpath(dir, "t.hds"), "w") do io
            for (k, t) in enumerate((1.0, 2.0))
                MO.write_head_record(io, L, Float32.(heads[k]); kstp = 1, kper = k, pertim = 1.0, totim = t, inactive)
            end
        end
        flowja = collect(range(-1.5, 2.25; length = 10))
        sto = collect(range(0.0, 1.0; length = 24))
        rch = MO.BudgetList(Int32[1, 2], Int32[1, 2], [0.5, 0.25], [1.0 2.0])
        open(joinpath(dir, "t.cbc"), "w") do io
            for (k, t) in enumerate((1.0, 2.0))
                kw = (; kstp = 1, kper = k, delt = 1.0, pertim = 1.0, totim = t)
                MO.write_budget_array(io, "FLOW-JA-FACE", flowja; kw...)
                MO.write_budget_array(io, L, "STO-SS", sto; kw...)
                MO.write_budget_list(io, L, "RCH", rch; kw..., modelnam1 = "M", paknam1 = "M", modelnam2 = "M", paknam2 = "RCH-1", auxtxt = ["MULT"])
                MO.write_budget_list(io, L, "WEL", MO.BudgetList(Int32[], Int32[], Float64[]); kw..., modelnam1 = "M", paknam1 = "M", modelnam2 = "M", paknam2 = "WEL-1")
            end
        end
        script = joinpath(@__DIR__, "..", "flopy_read.py")
        out = read(`uv run --quiet --no-project --with flopy --with numpy python -I $script $dir`, String)
        res = Dict{String, Vector{String}}()
        for l in split(out, '\n'; keepempty = false)
            w = split(l)
            res[w[1]] = w[2:end]
        end
        num(k) = parse.(Float64, res[k])
        @test num("hds_times") == [1.0, 2.0]
        @test num("hds_kstpkper") == [0.0, 0.0, 0.0, 1.0]  # flopy's (kstp, kper) are 0-based
        @test num("hds_shape") == [2.0, 3.0, 4.0]
        expected = Float64.(Float32.(heads[2]))
        expected[1] = MO.HNOFLO
        @test num("hds_data") == expected
        @test res["cbc_names"] == ["FLOW-JA-FACE", "STO-SS", "RCH", "WEL"]
        @test num("cbc_times") == [1.0, 2.0]
        @test num("rch_node") == [1.0, 2.0] && num("rch_node2") == [1.0, 2.0]
        @test num("rch_q") == [0.5, 0.25] && num("rch_mult") == [1.0, 2.0]
        @test num("wel_n") == [0.0]
        @test num("fjf") == flowja
        @test num("sto_shape") == [2.0, 3.0, 4.0]
        @test num("sto") == sto
    end
end

# ---------------------------------------------------------------------------------------
# Reference runs of T0.3
# ---------------------------------------------------------------------------------------

# Half a unit in the last printed digit of a `value_to_string` value (Budget.f90):
# F17.4 below 1e12, 1PE17.4 (5 significant digits) from there on; plus round-off.
_print_tol(v) = (abs(v) >= 9.99999e11 ? 5.0e-5 * abs(v) : 5.0e-5) * 1.01 + 1.0e-12 * abs(v)

function _files_equal(a, b)
    filesize(a) == filesize(b) || return false
    return open(a) do ia
        open(b) do ib
            while !eof(ia)
                read(ia, 1 << 24) == read(ib, 1 << 24) || return false
            end
            return true
        end
    end
end

# Write `x` (read from `path`) with `writer` to a temporary file and compare the bytes.
function _rewrite_identical(writer, x, path)
    tmp = tempname()
    try
        return _files_equal(writer(tmp, x), path)
    finally
        rm(tmp; force = true)
    end
end

# FLOW-JA-FACE is antisymmetric: flowja(n→m) = -flowja(m→n).
function _flowja_antisymmetric(flowja, ia, ja)
    for n in 1:(length(ia) - 1), ij in (ia[n] + 1):(ia[n + 1] - 1)
        m = ja[ij]
        ji = findfirst(==(n), view(ja, (ia[m] + 1):(ia[m + 1] - 1)))
        ji === nothing && return false
        flowja[ij] == -flowja[ia[m] + ji] || return false
    end
    return true
end

# Check the head file, budget file and listing of one model of a reference run.
function _check_reference_model(rdir, mname, m, nam, stats; rewrite, spec)
    haskey(m, "grb_file") || return @info "No binary grid file for $mname in $rdir: skipped"
    g = MO.read_grb(joinpath(rdir, m["grb_file"]))
    layout = MO.OutputLayout(g)
    grid, dims = layout.kind, MO.budget_dims(layout)
    nodes = MO.nodesuser(layout)
    if spec !== nothing
        @test MO.OutputLayout(MO.model_spec(spec, mname).dis) == layout
        stats[:spec_layouts] += 1
    end
    idomain = haskey(g, "IDOMAIN") ? vec(g["IDOMAIN"]) : ones(Int32, nodes)
    @test lowercase(g.grid) == lowercase(m["grid"])
    @test length(idomain) == nodes
    rewrite && @test _rewrite_identical(MO.write_grb, g, joinpath(rdir, m["grb_file"]))
    stats[:grb] += 1

    if haskey(m, "head_file")
        path = joinpath(rdir, m["head_file"])
        hf = MO.open_head_file(path; grid)
        @test length(hf) >= 1
        t = MO.head_times(hf)
        @test issorted(t)
        shape = grid === :dis ? dims : grid === :disv ? (dims[1], dims[3]) : (nodes,)
        inactive = idomain .<= 0
        for i in 1:length(hf)
            h = MO.read_head(hf, i)
            @test size(h) == shape
            @test strip(hf.steps[i].text) == "HEAD"
            v = vec(h)
            @test all(==(MO.HNOFLO), v[inactive])
            @test all(x -> isfinite(x) && x != MO.HNOFLO, v[.!inactive])
            stats[:head_arrays] += 1
        end
        if rewrite
            @test _rewrite_identical(MO.write_head_file, hf, path)
            stats[:rewritten] += 1
        end
        stats[:hds] += 1
    end

    if haskey(m, "budget_file")
        path = joinpath(rdir, m["budget_file"])
        bf = MO.open_budget_file(path)
        nja = Int(g["NJA"])
        for (i, r) in enumerate(bf.records)
            d = MO.read_budget(bf, i)
            if r.imeth == 1 && strip(r.text) == "FLOW-JA-FACE"
                @test (r.ndim1, r.ndim2, r.ndim3, r.n) == (nja, 1, -1, nja)
                @test _flowja_antisymmetric(d, g["IA"], g["JA"])
            else
                @test (r.ndim1, r.ndim2, -r.ndim3) == dims
                if r.imeth == 1
                    @test length(d) == nodes
                else
                    @test all(in(1:nodes), d.id1)
                    @test all(isfinite, d.q)
                    @test size(d.aux) == (length(r.auxtxt), length(d.q))
                end
            end
            stats[:cbc_records] += 1
        end
        if rewrite
            @test _rewrite_identical(MO.write_budget_file, bf, path)
            stats[:rewritten] += 1
        end
        stats[:cbc] += 1
    end

    lst = joinpath(rdir, splitext(nam)[1] * ".lst")
    isfile(lst) || return @info "No listing file $lst: skipped"
    tables = MO.read_listing_budgets(lst)
    @test !isempty(tables)
    for b in tables
        @test b.title == "VOLUME"
        # Totals are the sums of the printed terms, within print precision.
        for (tot, f) in (
                (b.total_rate_in, t -> t.rate_in), (b.total_rate_out, t -> t.rate_out),
                (b.total_cum_in, t -> t.cum_in), (b.total_cum_out, t -> t.cum_out),
            )
            @test abs(sum(f, b.terms; init = 0.0) - tot) <= _print_tol(tot) + sum(t -> _print_tol(f(t)), b.terms; init = 0.0)
        end
        @test abs(b.total_rate_in - b.total_rate_out - b.rate_in_minus_out) <=
            _print_tol(b.total_rate_in) + _print_tol(b.total_rate_out) + _print_tol(b.rate_in_minus_out)
        # The printed percent discrepancy (F15.2) agrees with the one computed from the
        # printed totals; the bound adds the effect of the rounding of the totals.
        for (tin, tout, pd) in ((b.total_rate_in, b.total_rate_out, b.rate_percent_discrepancy), (b.total_cum_in, b.total_cum_out, b.cum_percent_discrepancy))
            avg = (tin + tout) / 2
            bound = 0.005 + (avg > 0 ? 100 * (_print_tol(tin) + _print_tol(tout)) / avg * 2 : 0.0)
            @test abs(MO.percent_discrepancy(tin, tout) - pd) <= bound
        end
        stats[:tables] += 1
    end

    # Package flows in the budget file vs the rates in the listing file.
    haskey(m, "budget_file") || return nothing
    bf = MO.open_budget_file(joinpath(rdir, m["budget_file"]))
    for b in tables
        exchange_chd = Dict{String, Float64}()
        for t in b.terms
            idx = if startswith(t.name, "STO-")
                MO.find_budget_records(bf; text = t.name, kstp = b.kstp, kper = b.kper, imeth = 1)
            else
                MO.find_budget_records(bf; text = t.name, kstp = b.kstp, kper = b.kper, imeth = 6, paknam2 = t.package)
            end
            isempty(idx) && continue  # not saved (no SAVE_FLOWS, or not this time step)
            q = mapreduce(i -> (d = MO.read_budget(bf, i); d isa MO.BudgetList ? d.q : d), vcat, idx)
            qin = sum(x -> max(x, 0.0), q; init = 0.0)
            qout = -sum(x -> min(x, 0.0), q; init = 0.0)
            if t.name == "CHD"
                continue  # checked as a net flow below
            elseif t.name == "FLOW-JA-FACE"
                # A GWF-GWF exchange: MF6 lists the flows to constant-head cells of the
                # model as FLOW-JA-FACE-CHD, but saves all exchange flows in one record.
                chd = only(filter(c -> c.name == "FLOW-JA-FACE-CHD" && c.package == t.package, b.terms))
                @test abs(qin - t.rate_in - chd.rate_in) <= _print_tol(t.rate_in) + _print_tol(chd.rate_in)
                @test abs(qout - t.rate_out - chd.rate_out) <= _print_tol(t.rate_out) + _print_tol(chd.rate_out)
            else
                @test abs(qin - t.rate_in) <= _print_tol(t.rate_in)
                @test abs(qout - t.rate_out) <= _print_tol(t.rate_out)
            end
            stats[:matched_terms] += 1
        end
        # CHD: the listing sums the flows of each connection to an active cell
        # (ratechdin/ratechdout of `calc_chd_rate`, gwf-chd.f90), the budget file holds
        # the net flow of each CHD cell (connections between CHD cells cancel in the sum
        # over all CHD packages).
        chd = filter(t -> t.name == "CHD", b.terms)
        idx = MO.find_budget_records(bf; text = "CHD", kstp = b.kstp, kper = b.kper, imeth = 6)
        if !isempty(chd) && length(idx) == length(chd)
            net = sum(i -> sum(MO.read_budget(bf, i).q; init = 0.0), idx)
            @test abs(net - sum(t -> t.rate_in - t.rate_out, chd)) <= sum(t -> _print_tol(t.rate_in) + _print_tol(t.rate_out), chd)
            stats[:matched_terms] += length(chd)
        end
    end
    return nothing
end

@testset "output: MF6 reference runs" begin
    runs = joinpath(REFERENCE_DIR, "runs")
    summary = joinpath(runs, "summary.toml")
    if !isfile(summary)
        @info "Skipping the output-file checks on the reference runs: $summary not found (validation/run_reference.jl)"
    else
        inventory = TOML.parsefile(joinpath(@__DIR__, "..", "..", "validation", "examples.toml"))["examples"]
        per_tier = Dict{Int, Dict{Symbol, Int}}()
        for run in TOML.parsefile(summary)["runs"]
            name = run["name"]
            haskey(inventory, name) || continue
            ex = inventory[name]
            stats = get!(() -> Dict(k => 0 for k in (:runs, :grb, :hds, :head_arrays, :cbc, :cbc_records, :tables, :matched_terms, :rewritten, :spec_layouts)), per_tier, ex["tier"])
            # Every mode run by validation/run_reference.jl (raw, tight, tight-relative).
            modes = isdir(joinpath(runs, name)) ? filter(d -> isdir(joinpath(runs, name, d)), readdir(joinpath(runs, name))) : String[]
            isempty(modes) && @info "No reference run of $name: skipped"
            for mode in modes
                rdir = joinpath(runs, name, mode)
                # Runs that stopped early (no convergence) are read too: MF6 writes whole
                # records, so their files are complete up to the last saved time step.
                @testset "$name/$mode" begin
                    # The layout of the loaded DIS/DISV/DISU spec (from the unpatched
                    # example input) equals the .grb layout, checked on the reference mode.
                    spec = nothing
                    exdir = joinpath(REFERENCE_DIR, "mf6examples", ex["path"])
                    if mode == run["reference_mode"] && isdir(exdir)
                        spec = try
                            MO.load_simulation(exdir; strict = false)
                        catch err
                            @info "Simulation loader failed on $exdir: layout check skipped ($(sprint(showerror, err)))"
                            nothing
                        end
                    end
                    sim = MO.read_input_file(joinpath(rdir, "mfsim.nam"); ftype = "SIM", component = "SIM", simdir = rdir)
                    for l in MO.find_block(sim, "MODELS").lines
                        MO.keyword(l, 1) == "GWF6" || continue
                        mname = String(l[3])
                        key = only(filter(k -> lowercase(k) == lowercase(mname), keys(ex["models"])))
                        _check_reference_model(rdir, mname, ex["models"][key], String(l[2]), stats; rewrite = mode == run["reference_mode"], spec)
                    end
                end
                stats[:runs] += 1
            end
        end
        for tier in sort!(collect(keys(per_tier)))
            s = per_tier[tier]
            @info "Output files of the reference runs, tier $tier: $(s[:runs]) runs, $(s[:hds]) head files ($(s[:head_arrays]) arrays), " *
                "$(s[:cbc]) budget files ($(s[:cbc_records]) records), $(s[:grb]) grid files, $(s[:tables]) listing budget tables, " *
                "$(s[:matched_terms]) listing terms matched to budget records, $(s[:rewritten]) files rewritten byte-identically, " *
                "$(s[:spec_layouts]) spec layouts checked"
        end
    end
end
