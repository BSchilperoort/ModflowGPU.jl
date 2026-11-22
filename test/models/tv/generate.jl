# Writes the TVK6/TVS6 test models of T6.12 into test/models/tv/<case>/ (run with mf6 at
# test time, group tv):
#
# - tvk01, tvk02, tvk02_disv, tvk03, tvk04, tvk05, tvk06ts: the MF6 autotests
#   test_gwf_npf_tvk01.py ... tvk06.py (K, K22/K33 changes per period on DIS and DISV,
#   K33 not given in NPF, a TS6 series of K33); tvk02_negk (a K <= 0) is the error test of
#   test/unit/tv_test.jl;
# - tvs01 ... tvs05: test_gwf_sto_tvs01.py (one convertible cell, NEWTON, SY and SS changes;
#   tvs02/tvs05 with a STEPWISE TS6 series, tvs03 with DISABLE_STORAGE_CHANGE_INTEGRATION,
#   tvs04/tvs05 with one STO PERIOD block);
# - tv_conv: a convertible DIS model (2 × 8 × 10, layer 1 ICELLTYPE 1 / ICONVERT 1, layer 2
#   confined; K22 and K33 with K33OVERK) with transient periods of several steps: TVK with a
#   LINEAR TS6 series (K22 changing every step, a literal replacing a linked value later),
#   TVS without time series (SS of confined and convertible cells, SY). The NEWTON and
#   DISABLE_STORAGE_CHANGE_INTEGRATION variants are made at test time.
#
# Deviations from the autotests: the TVK models use OUTER_DVCLOSE 1e-9 and INNER_DVCLOSE /
# INNER_RCLOSE 1e-10 (the autotests' 1e-6 leaves differences of that size between two
# converged solvers), and save heads and budgets every step.
#
#   julia test/models/tv/generate.jl
const ROOT = @__DIR__

function write_case(case, files::Dict{String, String})
    dir = joinpath(ROOT, case)
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    for (f, t) in files
        write(joinpath(dir, f), t)
    end
    return dir
end

mfsim(name; nper, perioddata, ims = "") = Dict(
    "mfsim.nam" => """
    BEGIN options
    END options

    BEGIN timing
      TDIS6  $name.tdis
    END timing

    BEGIN models
      gwf6  $name.nam  $name
    END models

    BEGIN exchanges
    END exchanges

    BEGIN solutiongroup  1
      ims6  $name.ims  $name
    END solutiongroup  1
    """,
    "$name.tdis" => """
    BEGIN options
      TIME_UNITS  days
    END options

    BEGIN dimensions
      NPER  $nper
    END dimensions

    BEGIN perioddata
    $(join(("  " * p for p in perioddata), "\n"))
    END perioddata
    """,
    "$name.ims" => isempty(ims) ? """
    BEGIN options
      PRINT_OPTION  SUMMARY
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-9
      OUTER_MAXIMUM  100
      UNDER_RELAXATION  NONE
    END nonlinear

    BEGIN linear
      INNER_MAXIMUM  300
      INNER_DVCLOSE  1.0e-10
      INNER_RCLOSE  1.0e-10
      LINEAR_ACCELERATION  CG
      SCALING_METHOD  NONE
      REORDERING_METHOD  NONE
      RELAXATION_FACTOR  1.0
    END linear
    """ : ims,
)

nam(name, pkgs; options = "SAVE_FLOWS") = """
BEGIN options
  $options
END options

BEGIN packages
$(join(("  " * p for p in pkgs), "\n"))
END packages
"""

oc(name) = """
BEGIN options
  BUDGET  FILEOUT  $name.cbc
  HEAD  FILEOUT  $name.hds
END options

BEGIN period  1
  SAVE  HEAD  ALL
  SAVE  BUDGET  ALL
  PRINT  BUDGET  ALL
END period  1
"""

dis(nlay, nrow, ncol; top, botm, delr = 1.0, delc = 1.0) = """
BEGIN options
END options

BEGIN dimensions
  NLAY  $nlay
  NROW  $nrow
  NCOL  $ncol
END dimensions

BEGIN griddata
  delr
    CONSTANT  $delr
  delc
    CONSTANT  $delc
  top
    CONSTANT  $top
  botm  LAYERED
$(join(("    CONSTANT  $b" for b in botm), "\n"))
END griddata
"""

ic(strt::Real) = "BEGIN options\nEND options\n\nBEGIN griddata\n  strt\n    CONSTANT  $strt\nEND griddata\n"
ic(strt::Vector) = "BEGIN options\nEND options\n\nBEGIN griddata\n  strt\n    INTERNAL\n      $(join(strt, "  "))\nEND griddata\n"

function npf(; icelltype = 0, k, k22 = nothing, k33 = nothing, options = String[], tvk = nothing)
    opts = copy(options)
    isnothing(tvk) || push!(opts, "TVK6  FILEIN  $tvk")
    g = ["  icelltype\n    CONSTANT  $icelltype", "  k\n    CONSTANT  $k"]
    isnothing(k22) || push!(g, "  k22\n    CONSTANT  $k22")
    isnothing(k33) || push!(g, "  k33\n    CONSTANT  $k33")
    return "BEGIN options\n$(join(("  " * o for o in opts), "\n"))\nEND options\n\nBEGIN griddata\n$(join(g, "\n"))\nEND griddata\n"
end

periods(blocks) = join(("BEGIN period  $k\n$(join(("  " * r for r in recs), "\n"))\nEND period  $k\n" for (k, recs) in blocks), "\n")

list_pkg(blocks; maxbound, options = ["PRINT_FLOWS", "SAVE_FLOWS"]) = """
BEGIN options
$(join(("  " * o for o in options), "\n"))
END options

BEGIN dimensions
  MAXBOUND  $maxbound
END dimensions

$(periods(blocks))"""

tv(blocks; options = ["PRINT_INPUT"]) = "BEGIN options\n$(join(("  " * o for o in options), "\n"))\nEND options\n\n" * periods(blocks)

cid(c) = join(c, "  ")

# --- tvk01: 3 × 3 × 3, K of every cell 1 then 10 (two periods) -----------------------------
let name = "gwf_tvk01"
    cells = [(k, i, j) for k in 1:3 for i in 1:3 for j in 1:3]
    f = mfsim(name; nper = 2, perioddata = ["100.0  1  1.0", "100.0  1  1.0"])
    f["$name.nam"] = nam(name, ["DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "CHD6  $name.chd  chd-1", "OC6  $name.oc  oc"])
    f["$name.dis"] = dis(3, 3, 3; top = 1.0, botm = [0.0, -1.0, -2.0])
    f["$name.ic"] = ic(1.0)
    f["$name.npf"] = npf(; k = 0.1, k33 = 0.1, options = ["SAVE_SPECIFIC_DISCHARGE"], tvk = "$name.npf.tvk")
    f["$name.npf.tvk"] = tv([k => ["$(cid(c))  K  $(k == 1 ? 1.0 : 10.0)" for c in cells] for k in 1:2])
    chd = String[]
    for k in 1:3, i in 1:3
        push!(chd, "$k  $i  1  2.0", "$k  $i  3  1.0")
    end
    f["$name.chd"] = list_pkg([1 => chd]; maxbound = length(chd))
    f["$name.oc"] = oc(name)
    write_case("tvk01", f)
end

# --- tvk02 (DIS and DISV): 1 × 1 × 3, K of the end cells changing --------------------------
const TVK02 = [
    2 => [("first", "K", 3.0)], 3 => [("first", "K", 0.2)], 4 => [("first", "K", 1.0), ("last", "K", 3.0)],
    5 => [("last", "K", 0.2)], 7 => [("last", "K", 1.0)],
]
for (case, disv) in (("tvk02", false), ("tvk02_disv", true))
    name = "gwf_" * case
    f = mfsim(name; nper = 7, perioddata = fill("1.0  1  1.0", 7))
    dext = disv ? "disv" : "dis"
    f["$name.nam"] = nam(name, ["$(uppercase(dext))6  $name.$dext  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "CHD6  $name.chd  chd-1", "OC6  $name.oc  oc"])
    first, last = disv ? ((1, 1), (1, 3)) : ((1, 1, 1), (1, 1, 3))
    f["$name.$dext"] = disv ? """
    BEGIN options
    END options

    BEGIN dimensions
      NLAY  1
      NCPL  3
      NVERT  8
    END dimensions

    BEGIN griddata
      top
        CONSTANT  1.0
      botm
        CONSTANT  -1.0
    END griddata

    BEGIN vertices
      1  0.0  0.0
      2  1.0  0.0
      3  2.0  0.0
      4  3.0  0.0
      5  0.0  1.0
      6  1.0  1.0
      7  2.0  1.0
      8  3.0  1.0
    END vertices

    BEGIN cell2d
      1  0.5  0.5  4  5  6  2  1
      2  1.5  0.5  4  6  7  3  2
      3  2.5  0.5  4  7  8  4  3
    END cell2d
    """ : dis(1, 1, 3; top = 1.0, botm = [-1.0])
    f["$name.ic"] = ic([0.0, 1.0, 1.0])
    f["$name.npf"] = npf(; k = 1.0, k33 = 1.0, tvk = "$name.npf.tvk")
    f["$name.npf.tvk"] = tv([k => ["$(cid(w == "first" ? first : last))  $v  $x" for (w, v, x) in recs] for (k, recs) in TVK02])
    f["$name.chd"] = list_pkg([1 => ["$(cid(first))  0.0", "$(cid(last))  1.0"]]; maxbound = 2)
    f["$name.oc"] = oc(name)
    write_case(case, f)
end

# --- tvk03: 3 × 1 × 1, K33 of the end cells changing ---------------------------------------
let name = "gwf_tvk03"
    f = mfsim(name; nper = 7, perioddata = fill("1.0  1  1.0", 7))
    f["$name.nam"] = nam(name, ["DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "CHD6  $name.chd  chd-1", "OC6  $name.oc  oc"])
    f["$name.dis"] = dis(3, 1, 1; top = 0.0, botm = [-1.0, -2.0, -3.0])
    f["$name.ic"] = "BEGIN options\nEND options\n\nBEGIN griddata\n  strt  LAYERED\n    CONSTANT  1.0\n    CONSTANT  1.0\n    CONSTANT  0.0\nEND griddata\n"
    f["$name.npf"] = npf(; k = 1.0, k33 = 1.0, tvk = "$name.npf.tvk")
    c1, c3 = "1  1  1", "3  1  1"
    f["$name.npf.tvk"] = tv(
        [
            2 => ["$c3  K33  3.0"], 3 => ["$c3  K33  0.2"], 4 => ["$c3  K33  1.0", "$c1  K33  3.0"], 5 => ["$c1  K33  0.2"],
            7 => ["$c1  K33  1.0"],
        ],
    )
    f["$name.chd"] = list_pkg([1 => ["$c1  1.0", "$c3  0.0"]]; maxbound = 2)
    f["$name.oc"] = oc(name)
    write_case("tvk03", f)
end

# --- tvk04 / tvk05: K33 not given in NPF, set by TVK (3 layers / 1 layer), GHB -------------
for (case, nlay, k33, ghbcond) in (("tvk04", 3, 10.0, 10.0 * 1.0 * 1.0 / 0.5), ("tvk05", 1, 1000.0, 0.1 * 1.0 * 1.0 / 0.5))
    name = "gwf_" * case
    f = mfsim(name; nper = 2, perioddata = ["100.0  1  1.0", "100.0  1  1.0"])
    f["$name.nam"] = nam(
        name, ["DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "CHD6  $name.chd  chd-1", "GHB6  $name.ghb  ghb-1", "OC6  $name.oc  oc"],
    )
    f["$name.dis"] = dis(nlay, 3, 3; top = 1.0, botm = [0.0, -1.0, -2.0][1:nlay])
    f["$name.ic"] = ic(1.0)
    f["$name.npf"] = npf(; k = 0.1, k22 = 0.1, options = ["SAVE_SPECIFIC_DISCHARGE"], tvk = "$name.npf.tvk")
    f["$name.npf.tvk"] = tv([2 => ["$k  $i  $j  K33  $k33" for k in 1:nlay for i in 1:3 for j in 1:3]])
    f["$name.chd"] = list_pkg([1 => ["1  $i  1  2.0" for i in 1:3]]; maxbound = 3)
    f["$name.ghb"] = list_pkg([1 => ["$nlay  $i  3  1.0  $ghbcond" for i in 1:3]]; maxbound = 3)
    f["$name.oc"] = oc(name)
    write_case(case, f)
end

# --- tvk06ts: K33 of the bottom cell from a LINEAR series K33(t) = 1 + t -------------------
let name = "gwf_tvk06ts"
    f = mfsim(name; nper = 3, perioddata = fill("1.0  1  1.0", 3))
    f["$name.nam"] = nam(name, ["DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "CHD6  $name.chd  chd-1", "OC6  $name.oc  oc"])
    f["$name.dis"] = dis(3, 1, 1; top = 0.0, botm = [-1.0, -2.0, -3.0])
    f["$name.ic"] = ic(0.5)
    f["$name.npf"] = npf(; k = 1.0, k33 = 1.0, tvk = "$name.npf.tvk")
    f["$name.npf.tvk"] = tv([1 => ["3  1  1  K33  k33_series"]]; options = ["PRINT_INPUT", "TS6  FILEIN  $name.npf.tvk.ts"])
    f["$name.npf.tvk.ts"] = """
    BEGIN attributes
      NAME  k33_series
      METHOD  LINEAR
    END attributes

    BEGIN timeseries
      0.0  1.0
      1.0  2.0
      2.0  3.0
      3.0  4.0
    END timeseries
    """
    f["$name.chd"] = list_pkg([1 => ["1  1  1  1.0", "3  1  1  0.0"]]; maxbound = 2)
    f["$name.oc"] = oc(name)
    write_case("tvk06ts", f)
end

# --- tvs01 ... tvs05: one convertible cell, NEWTON UNDER_RELAXATION, DBD, backtracking -----
const TVS_IMS = """
BEGIN options
  PRINT_OPTION  SUMMARY
END options

BEGIN nonlinear
  OUTER_DVCLOSE  1.0e-9
  OUTER_MAXIMUM  1000
  UNDER_RELAXATION  DBD
  UNDER_RELAXATION_THETA  0.7
  UNDER_RELAXATION_KAPPA  0.07
  UNDER_RELAXATION_GAMMA  0.1
  UNDER_RELAXATION_MOMENTUM  0.0
  BACKTRACKING_NUMBER  200
  BACKTRACKING_TOLERANCE  1.1
  BACKTRACKING_REDUCTION_FACTOR  0.2
  BACKTRACKING_RESIDUAL_LIMIT  1.0
END nonlinear

BEGIN linear
  INNER_MAXIMUM  100
  INNER_DVCLOSE  1.0e-9
  INNER_RCLOSE  1.0e-12
  LINEAR_ACCELERATION  BICGSTAB
  SCALING_METHOD  NONE
  REORDERING_METHOD  NONE
  RELAXATION_FACTOR  0.0
END linear
"""
for (idx, case) in enumerate(("tvs01", "tvs02", "tvs03", "tvs04", "tvs05"))
    name = "gwf_" * case
    ts = idx in (2, 5)
    single = idx in (4, 5)
    f = mfsim(name; nper = 5, perioddata = fill("1.0  1  1.0", 5), ims = TVS_IMS)
    f["$name.nam"] = nam(name, ["DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "STO6  $name.sto  sto", "OC6  $name.oc  oc"]; options = "NEWTON  UNDER_RELAXATION\n  SAVE_FLOWS")
    f["$name.dis"] = dis(1, 1, 1; top = 1.0, botm = [0.0])
    f["$name.ic"] = ic(0.8)
    f["$name.npf"] = npf(; icelltype = 1, k = 1.0, k33 = 1.0)
    sto_periods = single ? "BEGIN period  1\n  TRANSIENT\nEND period  1\n" : join(("BEGIN period  $k\n  TRANSIENT\nEND period  $k\n" for k in 1:5), "\n")
    f["$name.sto"] = """
    BEGIN options
      TVS6  FILEIN  $name.sto.tvs
    END options

    BEGIN griddata
      iconvert
        CONSTANT  1
      ss
        CONSTANT  1.0e-6
      sy
        CONSTANT  0.01
    END griddata

    """ * sto_periods
    sy(v) = ts ? "1  1  1  SY  syh_series" : "1  1  1  SY  $v"
    opts = ["PRINT_INPUT"]
    idx == 3 && pushfirst!(opts, "DISABLE_STORAGE_CHANGE_INTEGRATION")
    ts && push!(opts, "TS6  FILEIN  $name.ts")
    f["$name.sto.tvs"] = tv([2 => [sy(0.005)], 3 => ["1  1  1  SS  1.0e-5"], 4 => [sy(0.02)], 5 => ["1  1  1  SS  1.0e-6", sy(0.01)]]; options = opts)
    ts && (
        f["$name.ts"] = """
        BEGIN attributes
          NAME  syh_series
          METHOD  STEPWISE
        END attributes

        BEGIN timeseries
          1.0  0.005
          3.0  0.02
          4.0  0.01
          5.0  0.01
        END timeseries
        """
    )
    f["$name.oc"] = oc(name)
    write_case(case, f)
end

# --- tv_conv: convertible DIS model with TVK (time series) and TVS -------------------------
let name = "tvc", nrow = 8, ncol = 10
    f = mfsim(
        name; nper = 4, perioddata = ["1.0  1  1.0", "20.0  3  1.2", "20.0  4  1.0", "20.0  2  1.0"],
        ims = """
        BEGIN options
          PRINT_OPTION  SUMMARY
        END options

        BEGIN nonlinear
          OUTER_DVCLOSE  1.0e-9
          OUTER_MAXIMUM  500
        END nonlinear

        BEGIN linear
          INNER_MAXIMUM  500
          INNER_DVCLOSE  1.0e-10
          INNER_RCLOSE  1.0e-6
          LINEAR_ACCELERATION  CG
        END linear
        """,
    )
    f["$name.nam"] = nam(
        name, [
            "DIS6  $name.dis  dis", "IC6  $name.ic  ic", "NPF6  $name.npf  npf", "STO6  $name.sto  sto", "CHD6  $name.chd  chd-1",
            "WEL6  $name.wel  wel-1", "RCH6  $name.rch  rch-1", "OC6  $name.oc  oc",
        ],
    )
    top = [round(20.0 - 0.25 * j + 0.15 * mod(i, 3); digits = 2) for i in 1:nrow, j in 1:ncol]
    f["$name.dis"] = """
    BEGIN options
    END options

    BEGIN dimensions
      NLAY  2
      NROW  $nrow
      NCOL  $ncol
    END dimensions

    BEGIN griddata
      delr
        CONSTANT  50.0
      delc
        CONSTANT  40.0
      top
        INTERNAL
    $(join(("      " * join(top[i, :], "  ") for i in 1:nrow), "\n"))
      botm  LAYERED
        CONSTANT  8.0
        CONSTANT  -10.0
    END griddata
    """
    f["$name.ic"] = "BEGIN options\nEND options\n\nBEGIN griddata\n  strt  LAYERED\n    CONSTANT  15.0\n    CONSTANT  15.0\nEND griddata\n"
    f["$name.npf"] = """
    BEGIN options
      SAVE_FLOWS
      K33OVERK
      TVK6  FILEIN  $name.npf.tvk
    END options

    BEGIN griddata
      icelltype  LAYERED
        CONSTANT  1
        CONSTANT  0
      k  LAYERED
        CONSTANT  12.0
        CONSTANT  4.0
      k22  LAYERED
        CONSTANT  6.0
        CONSTANT  4.0
      k33  LAYERED
        CONSTANT  0.1
        CONSTANT  0.2
    END griddata
    """
    # Period 1: K of a block of layer-1 cells, a K33 ratio; period 2: K22 of a column of
    # layer 1 from a LINEAR series (every step), K of layer-2 cells; period 4: a literal
    # replaces the series of one cell, K back.
    f["$name.npf.tvk"] = tv(
        [
            1 => ["1  $i  $j  K  30.0" for i in 3:5 for j in 4:6] ∪ ["2  4  5  K33  0.02"],
            2 => vcat(["1  $i  7  K22  kser" for i in 1:nrow], ["2  $i  3  K  1.5" for i in 2:6]),
            4 => ["1  2  7  K22  0.5", "1  4  5  K  12.0", "2  3  3  K33  0.5"],
        ]; options = ["PRINT_INPUT", "TS6  FILEIN  $name.npf.tvk.ts"],
    )
    f["$name.npf.tvk.ts"] = """
    BEGIN attributes
      NAME  kser
      METHOD  LINEAR
    END attributes

    BEGIN timeseries
      0.0  6.0
      21.0  2.0
      41.0  9.0
      61.0  0.8
    END timeseries
    """
    f["$name.sto"] = """
    BEGIN options
      SAVE_FLOWS
      TVS6  FILEIN  $name.sto.tvs
    END options

    BEGIN griddata
      iconvert  LAYERED
        CONSTANT  1
        CONSTANT  0
      ss
        CONSTANT  1.0e-5
      sy
        CONSTANT  0.15
    END griddata

    BEGIN period  1
      STEADY-STATE
    END period  1

    BEGIN period  2
      TRANSIENT
    END period  2
    """
    # Period 2: SS of confined layer-2 cells and SY of layer-1 cells; period 3: SS of
    # convertible cells, SY back for some; period 4: SS of layer 2 back.
    f["$name.sto.tvs"] = tv(
        [
            2 => vcat(["2  $i  $j  SS  4.0e-4" for i in 2:4 for j in 2:8], ["1  $i  $j  SY  0.05" for i in 5:7 for j in 3:6]),
            3 => vcat(["1  $i  $j  SS  2.0e-4" for i in 1:3 for j in 5:9], ["1  6  $j  SY  0.15" for j in 3:6]),
            4 => ["2  $i  $j  SS  1.0e-5" for i in 2:4 for j in 2:8],
        ],
    )
    f["$name.chd"] = list_pkg([1 => ["1  $i  1  15.5" for i in 1:nrow] ∪ ["1  $i  $ncol  12.5" for i in 1:nrow]]; maxbound = 2 * nrow)
    f["$name.wel"] = list_pkg([1 => ["2  4  4  0.0"], 2 => ["2  4  4  -400.0", "2  6  8  -250.0"], 4 => ["2  4  4  -100.0"]]; maxbound = 2)
    f["$name.rch"] = list_pkg([1 => ["1  $i  $j  2.0e-3" for i in 2:7 for j in 2:9]]; maxbound = 48)
    f["$name.oc"] = oc(name)
    write_case("tv_conv", f)
end
