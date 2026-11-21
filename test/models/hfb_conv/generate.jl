# Writes test/models/hfb_conv (T6.11): a convertible DIS model whose barriers change between
# stress periods. DIS 2 × 10 × 12 (uneven DELR, tops sloping), layer 1 convertible
# (ICELLTYPE 1, ICONVERT 1) with the water table inside it, layer 2 confined; CHD in the first
# and last columns of layer 1, wells in layer 2, recharge. Barriers:
# - period 1: a line across the model in layer 1 (horizontal connections with convertible
#   cells: `hfb_fc` every formulation), a shorter line in layer 2 (both cells confined:
#   `condsat_modify`), vertical barriers between the layers (a convertible cell: `hfb_fc`)
#   and a north-south line in layer 2 with a negative HYDCHR (a multiplier);
# - period 2: no block (MF6 reuses the barriers);
# - period 3: a new set (a line along a row in layer 1, HYDCHR 0 in layer 2);
# - period 4: an empty block (no barriers).
# Period 1 is steady state, the others transient. The NEWTON variant is made at test time
# (group hfb), where every barrier changes the saturated conductance. Run with mf6 at test time.
#
#   julia test/models/hfb_conv/generate.jl
const DIR = @__DIR__
const NAME = "hfbc"
const NLAY, NROW, NCOL = 2, 10, 12

w(name, txt) = write(joinpath(DIR, name), txt)
lines(v) = join(("  " * l for l in v), "\n")

delr = [40.0 + 5.0 * mod(j, 3) for j in 1:NCOL]
top = [round(20.0 - 0.3 * j + 0.2 * mod(i, 2); digits = 2) for i in 1:NROW, j in 1:NCOL]

w(
    "mfsim.nam", """
    # Convertible DIS model with HFB6 barriers changing between periods (T6.11).
    BEGIN options
    END options

    BEGIN timing
      TDIS6  $NAME.tdis
    END timing

    BEGIN models
      gwf6  $NAME.nam  $NAME
    END models

    BEGIN exchanges
    END exchanges

    BEGIN solutiongroup  1
      ims6  $NAME.ims  $NAME
    END solutiongroup  1
    """,
)
w(
    "$NAME.tdis", """
    BEGIN options
      TIME_UNITS  days
    END options

    BEGIN dimensions
      NPER  4
    END dimensions

    BEGIN perioddata
      1.0  1  1.0
      30.0  3  1.2
      30.0  3  1.2
      30.0  2  1.0
    END perioddata
    """,
)
w(
    "$NAME.ims", """
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
    "$NAME.nam", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN packages
      DIS6  $NAME.dis  dis
      NPF6  $NAME.npf  npf
      IC6  $NAME.ic  ic
      STO6  $NAME.sto  sto
      HFB6  $NAME.hfb  hfb
      CHD6  $NAME.chd  chd
      WEL6  $NAME.wel  wel
      RCH6  $NAME.rch  rch
      OC6  $NAME.oc  oc
    END packages
    """,
)
w(
    "$NAME.dis", """
    BEGIN options
      LENGTH_UNITS  meters
    END options

    BEGIN dimensions
      NLAY  $NLAY
      NROW  $NROW
      NCOL  $NCOL
    END dimensions

    BEGIN griddata
      delr
        INTERNAL  FACTOR  1.0
          $(join(string.(delr), "  "))
      delc
        CONSTANT  50.0
      top
        INTERNAL  FACTOR  1.0
    $(lines(["    " * join(string.(top[i, :]), "  ") for i in 1:NROW]))
      botm  LAYERED
        CONSTANT  5.0
        CONSTANT  -20.0
    END griddata
    """,
)
w(
    "$NAME.npf", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      icelltype  LAYERED
        CONSTANT  1
        CONSTANT  0
      k  LAYERED
        CONSTANT  10.0
        CONSTANT  5.0
      k33  LAYERED
        CONSTANT  1.0
        CONSTANT  0.5
    END griddata
    """,
)
w(
    "$NAME.ic", """
    BEGIN griddata
      strt
        CONSTANT  14.0
    END griddata
    """,
)
w(
    "$NAME.sto", """
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
w(
    "$NAME.chd", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXBOUND  $(2NROW)
    END dimensions

    BEGIN period  1
    $(lines(["1  $i  1  15.0" for i in 1:NROW]))
    $(lines(["1  $i  $NCOL  10.0" for i in 1:NROW]))
    END period  1
    """,
)
w(
    "$NAME.wel", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXBOUND  2
    END dimensions

    BEGIN period  1
      2  3  4  -150.0
    END period  1

    BEGIN period  2
      2  3  4  -400.0
      2  8  9  -300.0
    END period  2
    """,
)
w(
    "$NAME.rch", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXBOUND  $(NROW * (NCOL - 2))
    END dimensions

    BEGIN period  1
    $(lines(["1  $i  $j  0.002" for i in 1:NROW for j in 2:(NCOL - 1)]))
    END period  1
    """,
)
p1 = [
    ["1  $i  6  1  $i  7  1.0e-3" for i in 1:NROW];
    ["2  $i  6  2  $i  7  1.0e-4" for i in 3:8];
    ["1  $i  $j  2  $i  $j  2.0e-2" for (i, j) in ((2, 3), (2, 4), (9, 10), (5, 11))];
    ["2  5  $j  2  6  $j  -0.1" for j in 2:4];
]
p3 = [
    ["1  5  $j  1  6  $j  5.0e-4" for j in 2:(NCOL - 1)];
    ["2  $i  9  2  $i  10  0.0" for i in 1:4];
]
w(
    "$NAME.hfb", """
    BEGIN options
      PRINT_INPUT
    END options

    BEGIN dimensions
      MAXHFB  $(max(length(p1), length(p3)))
    END dimensions

    BEGIN period  1
    $(lines(p1))
    END period  1

    BEGIN period  3
    $(lines(p3))
    END period  3

    BEGIN period  4
    END period  4
    """,
)
w(
    "$NAME.oc", """
    BEGIN options
      BUDGET  FILEOUT  $NAME.cbc
      HEAD  FILEOUT  $NAME.hds
    END options

    BEGIN period  1
      SAVE  HEAD  ALL
      SAVE  BUDGET  ALL
      PRINT  BUDGET  ALL
    END period  1
    """,
)
