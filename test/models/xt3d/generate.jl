# Writes the synthetic XT3D model of test/models/xt3d (T6.5): DIS 3 × 6 × 7 with uneven layer
# bottoms and one excluded cell (IDOMAIN 0), full K tensor (K, K22, K33, ANGLE1, ANGLE2,
# ANGLE3: the horizontal connections have a z component), CHD, WEL, GHB, RCH, STO (steady
# period, then a transient one of 3 steps). Run with mf6 at test time (group xt3d).
#
#   julia test/models/xt3d/generate.jl
const DIR = @__DIR__
const NLAY, NROW, NCOL = 3, 6, 7

lcg = Ref(UInt64(12345))
rnd() = (lcg[] = lcg[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64((lcg[] >> 33) % 1000) / 1000)

function arr(io, name, f; layered = true)
    println(io, "  $name  LAYERED")
    for k in 1:NLAY
        println(io, "    INTERNAL  FACTOR  1.0")
        for i in 1:NROW
            println(io, "      ", join((string(round(f(k, i, j); digits = 4)) for j in 1:NCOL), "  "))
        end
    end
    return nothing
end

w(name, txt) = write(joinpath(DIR, name), txt)

w(
    "mfsim.nam", """
    # Synthetic XT3D model (T6.5): full K tensor with ANGLE1-3, uneven layers, CHD, WEL, GHB, RCH, STO.
    BEGIN options
    END options

    BEGIN timing
      TDIS6  xt3d.tdis
    END timing

    BEGIN models
      gwf6  xt3d.nam  xt3d
    END models

    BEGIN exchanges
    END exchanges

    BEGIN solutiongroup  1
      ims6  xt3d.ims  xt3d
    END solutiongroup  1
    """,
)
w(
    "xt3d.tdis", """
    BEGIN options
      TIME_UNITS  days
    END options

    BEGIN dimensions
      NPER  2
    END dimensions

    BEGIN perioddata
      1.0  1  1.0
      30.0  3  1.2
    END perioddata
    """,
)
w(
    "xt3d.ims", """
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
      INNER_RCLOSE  1.0e-10
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """,
)
w(
    "xt3d.nam", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN packages
      DIS6  xt3d.dis  dis
      NPF6  xt3d.npf  npf
      IC6  xt3d.ic  ic
      STO6  xt3d.sto  sto
      CHD6  xt3d.chd  chd
      WEL6  xt3d.wel  wel
      GHB6  xt3d.ghb  ghb
      RCH6  xt3d.rch  rch
      OC6  xt3d.oc  oc
    END packages
    """,
)
open(joinpath(DIR, "xt3d.dis"), "w") do io
    println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options\n")
    println(io, "BEGIN dimensions\n  NLAY  $NLAY\n  NROW  $NROW\n  NCOL  $NCOL\nEND dimensions\n")
    println(io, "BEGIN griddata")
    println(io, "  delr\n    INTERNAL  FACTOR  1.0\n      ", join((string(80.0 + 10 * (j % 3)) for j in 1:NCOL), "  "))
    println(io, "  delc\n    INTERNAL  FACTOR  1.0\n      ", join((string(70.0 + 15 * (i % 2)) for i in 1:NROW), "  "))
    println(io, "  top\n    INTERNAL  FACTOR  1.0")
    for i in 1:NROW
        println(io, "      ", join((string(round(50.0 + 2 * rnd(); digits = 4)) for j in 1:NCOL), "  "))
    end
    arr(io, "botm", (k, i, j) -> (k == 1 ? 30.0 : k == 2 ? 10.0 : -15.0) + 4 * rnd() - 0.5 * j)
    println(io, "  idomain  LAYERED")
    for k in 1:NLAY
        println(io, "    INTERNAL  FACTOR  1")
        for i in 1:NROW
            println(io, "      ", join((k == 2 && i == 3 && j == 5 ? "0" : "1" for j in 1:NCOL), "  "))
        end
    end
    println(io, "END griddata")
end
open(joinpath(DIR, "xt3d.npf"), "w") do io
    println(io, "BEGIN options\n  XT3D\n  SAVE_SPECIFIC_DISCHARGE\nEND options\n")
    println(io, "BEGIN griddata\n  icelltype\n    CONSTANT  0")
    arr(io, "k", (k, i, j) -> 5.0 + 10 * rnd())
    arr(io, "k22", (k, i, j) -> 0.5 + 2 * rnd())
    arr(io, "k33", (k, i, j) -> 0.1 + 0.5 * rnd())
    arr(io, "angle1", (k, i, j) -> 30.0 + 20 * rnd() - 10 * k)
    arr(io, "angle2", (k, i, j) -> 5.0 * rnd() - 2.5)
    arr(io, "angle3", (k, i, j) -> 10.0 * rnd() - 5.0)
    println(io, "END griddata")
end
w("xt3d.ic", "BEGIN griddata\n  strt\n    CONSTANT  45.0\nEND griddata\n")
w(
    "xt3d.sto", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      iconvert
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
open(joinpath(DIR, "xt3d.chd"), "w") do io
    println(io, "BEGIN options\n  SAVE_FLOWS\nEND options\n")
    println(io, "BEGIN dimensions\n  MAXBOUND  $(2 * NROW)\nEND dimensions\n")
    println(io, "BEGIN period  1")
    for i in 1:NROW
        println(io, "  1 $i 1  $(48.0 - 0.2 * i)")
        println(io, "  3 $i $NCOL  $(44.0 + 0.1 * i)")
    end
    println(io, "END period  1")
end
w(
    "xt3d.wel", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN dimensions
      MAXBOUND  2
    END dimensions

    BEGIN period  2
      2 4 4  -800.0
      3 2 3  -300.0
    END period  2
    """,
)
w(
    "xt3d.ghb", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN dimensions
      MAXBOUND  3
    END dimensions

    BEGIN period  1
      1 6 3  46.0  50.0
      1 6 4  46.5  60.0
      1 1 6  45.5  40.0
    END period  1
    """,
)
w(
    "xt3d.rch", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN dimensions
      MAXBOUND  4
    END dimensions

    BEGIN period  1
      1 2 2  0.002
      1 3 3  0.003
      1 4 5  0.001
      1 5 6  0.002
    END period  1
    """,
)
w(
    "xt3d.oc", """
    BEGIN options
      BUDGET  FILEOUT  xt3d.cbc
      HEAD  FILEOUT  xt3d.hds
    END options

    BEGIN period  1
      SAVE  HEAD  ALL
      SAVE  BUDGET  ALL
    END period  1
    """,
)
