# Writes the XT3D test models of T6.14 (head-dependent XT3D, src/gwf/xt3d_var.jl) under
# test/models/xt3d_var:
#
# - `xt3dnr_cont`, `xt3dnr_conv`, `xt3dnr_conv_std`: the MF6 autotest
#   test_gwf_xt3d_newton_upstream.py (a 1 × 3 × 5 unconfined model with rotated anisotropy
#   and NEWTON; a 1 × 4 × 10 model whose rows have equal transmissivity but different
#   saturations, with NEWTON and with the standard formulation);
# - `xt3d_conv`: DIS 2 × 7 × 9, layer 1 convertible (ICELLTYPE 1, ICONVERT 1, uneven
#   bottoms, a raised bottom where cells dry), layer 2 confined, K, K22, K33, ANGLE1 varying,
#   one IDOMAIN 0 cell, REWET; a steady period, a pumping period that dries cells and a
#   recovery period that rewets them. The NEWTON variant is made at test time.
#
# Run with mf6 at test time (group xt3d_var).
#
#   julia test/models/xt3d_var/generate.jl
const DIR = @__DIR__

lcg = Ref(UInt64(271828))
rnd() = (lcg[] = lcg[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64((lcg[] >> 33) % 1000) / 1000)

function wsim(dir, name, nper, perioddata, ims; newton = "")
    mkpath(dir)
    w(f, t) = write(joinpath(dir, f), t)
    w(
        "mfsim.nam", """
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
    )
    w(
        "$name.tdis", """
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
    )
    w("$name.ims", ims)
    w(
        "$name.oc", """
        BEGIN options
          BUDGET  FILEOUT  $name.cbc
          HEAD  FILEOUT  $name.hds
        END options

        BEGIN period  1
          SAVE  HEAD  ALL
          SAVE  BUDGET  ALL
          PRINT  BUDGET  ALL
        END period  1
        """,
    )
    return w
end

dis(nlay, nrow, ncol, delr, delc, top, botm) = """
BEGIN options
  LENGTH_UNITS  meters
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
$botm
END griddata
"""

nam(name, packages; options = "") = """
BEGIN options
  SAVE_FLOWS
$options
END options

BEGIN packages
$(join(("  " * p for p in packages), "\n"))
END packages
"""

chd(records) = """
BEGIN options
  SAVE_FLOWS
END options

BEGIN dimensions
  MAXBOUND  $(length(records))
END dimensions

BEGIN period  1
$(join(("  " * r for r in records), "\n"))
END period  1
"""

grid(v) = join(("      " * join((string(x) for x in row), "  ") for row in eachrow(v)), "\n")

# --- test_gwf_xt3d_newton_upstream.py ---------------------------------------------------

let name = "xt3dnr_cont", nrow = 3, ncol = 5, href = 80.0
    ims = """
    BEGIN options
      PRINT_OPTION  SUMMARY
      NO_PTC  ALL
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-9
      OUTER_MAXIMUM  500
    END nonlinear

    BEGIN linear
      INNER_DVCLOSE  1.0e-10
      INNER_RCLOSE  1.0e-8  STRICT
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """
    w = wsim(joinpath(DIR, name), name, 1, ["1.0  1  1.0"], ims)
    botm = zeros(nrow, ncol)
    botm[2, 4] = 60.0
    w("$name.dis", dis(1, nrow, ncol, 100.0, 100.0, 100.0, "  botm\n    INTERNAL  FACTOR  1.0\n" * grid(botm)))
    w("$name.nam", nam(name, ["DIS6  $name.dis  dis", "NPF6  $name.npf  npf", "IC6  $name.ic  ic", "CHD6  $name.chd  chd", "OC6  $name.oc  oc"]; options = "  NEWTON  UNDER_RELAXATION"))
    w("$name.ic", "BEGIN griddata\n  strt\n    CONSTANT  $href\nEND griddata\n")
    w(
        "$name.npf", """
        BEGIN options
          XT3D
        END options

        BEGIN griddata
          icelltype
            CONSTANT  1
          k
            CONSTANT  1.0
          k22
            CONSTANT  0.1
          angle1
            CONSTANT  30.0
        END griddata
        """,
    )
    yc(i) = (nrow - i) * 100.0 + 50.0
    xc(j) = (j - 1) * 100.0 + 50.0
    hb(i, j) = href + 0.05 * (yc(i) - yc(2)) + 0.01 * (xc(j) - xc(3))
    recs = ["1  $i  $j  $(hb(i, j))" for i in 1:nrow for j in 1:ncol if i in (1, nrow) || j in (1, ncol)]
    w("$name.chd", chd(recs))
end

for (name, newton) in (("xt3dnr_conv", true), ("xt3dnr_conv_std", false))
    nrow, ncol = 4, 10
    ims = """
    BEGIN options
      PRINT_OPTION  SUMMARY
      COMPLEXITY  COMPLEX
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-6
      OUTER_MAXIMUM  100
    END nonlinear

    BEGIN linear
      INNER_DVCLOSE  1.0e-8
      INNER_RCLOSE  1.0e-6  STRICT
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """
    w = wsim(joinpath(DIR, name), name, 1, ["1.0  1  1.0"], ims)
    botm = zeros(nrow, ncol)
    botm[3, :] .= 80.0
    k = ones(nrow, ncol)
    k[3, :] .= 9.0
    w("$name.dis", dis(1, nrow, ncol, 100.0, 100.0, 100.0, "  botm\n    INTERNAL  FACTOR  1.0\n" * grid(botm)))
    w(
        "$name.nam",
        nam(name, ["DIS6  $name.dis  dis", "NPF6  $name.npf  npf", "IC6  $name.ic  ic", "CHD6  $name.chd  chd", "OC6  $name.oc  oc"]; options = newton ? "  NEWTON  UNDER_RELAXATION" : ""),
    )
    w("$name.ic", "BEGIN griddata\n  strt\n    CONSTANT  90.0\nEND griddata\n")
    w(
        "$name.npf", """
        BEGIN options
          XT3D
        END options

        BEGIN griddata
          icelltype
            CONSTANT  1
          k
            INTERNAL  FACTOR  1.0
        $(grid(k))
          k22
            CONSTANT  0.1
          angle1
            CONSTANT  30.0
        END griddata
        """,
    )
    w("$name.chd", chd([["1  $i  1  92.0" for i in 1:nrow]; ["1  $i  $ncol  88.0" for i in 1:nrow]]))
end

# --- xt3d_conv ------------------------------------------------------------------------

let name = "xt3d_conv", nlay = 2, nrow = 7, ncol = 9
    ims = """
    BEGIN options
      PRINT_OPTION  SUMMARY
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-8
      OUTER_MAXIMUM  500
    END nonlinear

    BEGIN linear
      INNER_MAXIMUM  300
      INNER_DVCLOSE  1.0e-10
      INNER_RCLOSE  1.0e-9
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """
    w = wsim(joinpath(DIR, name), name, 3, ["1.0  1  1.0", "20.0  4  1.2", "40.0  4  1.2"], ims)
    top = [round(20.0 + 0.5 * rnd(); digits = 3) for i in 1:nrow, j in 1:ncol]
    bot1 = [round(6.0 + 3.0 * rnd() + 0.2 * j; digits = 3) for i in 1:nrow, j in 1:ncol]
    # A raised bottom near the well: these cells dry when it pumps.
    for (i, j) in ((3, 5), (3, 6), (4, 5), (4, 6))
        bot1[i, j] = 13.2 + 0.1 * j
    end
    bot2 = fill(-20.0, nrow, ncol)
    idom2 = ones(Int, nrow, ncol)
    idom2[6, 3] = 0
    io = IOBuffer()
    println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options\n")
    println(io, "BEGIN dimensions\n  NLAY  $nlay\n  NROW  $nrow\n  NCOL  $ncol\nEND dimensions\n")
    println(io, "BEGIN griddata")
    println(io, "  delr\n    INTERNAL  FACTOR  1.0\n      ", join((string(90.0 + 10.0 * (j % 3)) for j in 1:ncol), "  "))
    println(io, "  delc\n    INTERNAL  FACTOR  1.0\n      ", join((string(80.0 + 15.0 * (i % 2)) for i in 1:nrow), "  "))
    println(io, "  top\n    INTERNAL  FACTOR  1.0\n", grid(top))
    println(io, "  botm  LAYERED\n    INTERNAL  FACTOR  1.0\n", grid(bot1), "\n    INTERNAL  FACTOR  1.0\n", grid(bot2))
    println(io, "  idomain  LAYERED\n    CONSTANT  1\n    INTERNAL  FACTOR  1\n", grid(idom2))
    println(io, "END griddata")
    w("$name.dis", String(take!(io)))
    w(
        "$name.nam",
        nam(
            name, [
                "DIS6  $name.dis  dis", "NPF6  $name.npf  npf", "IC6  $name.ic  ic", "STO6  $name.sto  sto",
                "CHD6  $name.chd  chd", "WEL6  $name.wel  wel", "RCH6  $name.rch  rch", "OC6  $name.oc  oc",
            ],
        ),
    )
    w("$name.ic", "BEGIN griddata\n  strt\n    CONSTANT  15.0\nEND griddata\n")
    kk = [round(4.0 + 8.0 * rnd(); digits = 3) for _ in 1:(nlay * nrow), _ in 1:ncol]
    k22 = [round(1.0 + 2.0 * rnd(); digits = 3) for _ in 1:(nlay * nrow), _ in 1:ncol]
    ang = [round(20.0 + 30.0 * rnd(); digits = 2) for _ in 1:(nlay * nrow), _ in 1:ncol]
    lay(v) = "    INTERNAL  FACTOR  1.0\n" * grid(v[1:nrow, :]) * "\n    INTERNAL  FACTOR  1.0\n" * grid(v[(nrow + 1):end, :])
    w(
        "$name.npf", """
        BEGIN options
          SAVE_FLOWS
          XT3D
          REWET  WETFCT  1.0  IWETIT  1  IHDWET  0
        END options

        BEGIN griddata
          icelltype  LAYERED
            CONSTANT  1
            CONSTANT  0
          k  LAYERED
        $(lay(kk))
          k22  LAYERED
        $(lay(k22))
          k33  LAYERED
            CONSTANT  0.5
            CONSTANT  0.2
          angle1  LAYERED
        $(lay(ang))
          wetdry  LAYERED
            CONSTANT  0.5
            CONSTANT  0.0
        END griddata
        """,
    )
    w(
        "$name.sto", """
        BEGIN options
          SAVE_FLOWS
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
        """,
    )
    w("$name.chd", chd([["$k  $i  1  15.0" for k in 1:nlay for i in 1:nrow]; ["$k  $i  $ncol  13.5" for k in 1:nlay for i in 1:nrow]]))
    w(
        "$name.wel", """
        BEGIN options
          SAVE_FLOWS
        END options

        BEGIN dimensions
          MAXBOUND  2
        END dimensions

        BEGIN period  2
          1  4  4  -300.0
          2  4  6  -400.0
        END period  2

        BEGIN period  3
        END period  3
        """,
    )
    w(
        "$name.rch", """
        BEGIN options
          SAVE_FLOWS
        END options

        BEGIN dimensions
          MAXBOUND  $(nrow * ncol)
        END dimensions

        BEGIN period  1
        $(join(("  1  $i  $j  $(round(5.0e-4 + 1.0e-3 * rnd(); digits = 6))" for i in 1:nrow for j in 1:ncol), "\n"))
        END period  1
        """,
    )
end
