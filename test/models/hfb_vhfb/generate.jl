# Writes test/models/hfb_vhfb (T6.11), the model of MF6's autotest test_gwf_vhfb.py (case
# vhfb00): DIS 2 × 11 × 11, convertible cells (ICELLTYPE 1), a barrier on every vertical
# connection between the two layers (hydchr = 0.001 / 50), CHD in the first and last column
# of layer 1, one well in layer 2. The Newton (vhfb01) and logarithmic averaging (vhfb04)
# cases are made from it at test time (group hfb). Run with mf6 at test time.
# Changes from the autotest: K = 10 (autotest 100) and INNER_RCLOSE 1e-8. With K = 100 the
# horizontal conductances (5000) are 25000 times the barrier conductances (0.2), and the
# Float32 rounding of the offset heads moves the `:single` heads of layer 2 by up to 1.0e-3 m
# (the criterion); with K = 10, 8e-5 m. MF6 does not converge with INNER_RCLOSE 1e-10 here
# (the inner solves stop at INNER_MAXIMUM at the rounding level of the residual).
# The group hfb also runs the autotest's K = 100 in `:double` (`hfb_vhfb_k100`).
#
#   julia test/models/hfb_vhfb/generate.jl
const DIR = @__DIR__
const NAME = "vhfb"
const NLAY, NROW, NCOL = 2, 11, 11

w(name, txt) = write(joinpath(DIR, name), txt)

w(
    "mfsim.nam", """
    # MF6 autotest test_gwf_vhfb.py, case vhfb00 (T6.11): vertical HFB between convertible layers.
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
      NPER  1
    END dimensions

    BEGIN perioddata
      2.0  1  1.0
    END perioddata
    """,
)
w(
    "$NAME.ims", """
    BEGIN options
      PRINT_OPTION  SUMMARY
      COMPLEXITY  SIMPLE
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-9
      OUTER_MAXIMUM  200
    END nonlinear

    BEGIN linear
      INNER_MAXIMUM  300
      INNER_DVCLOSE  1.0e-10
      INNER_RCLOSE  1.0e-8
      LINEAR_ACCELERATION  CG
    END linear
    """,
)
w(
    "$NAME.nam", """
    BEGIN options
      PRINT_INPUT
    END options

    BEGIN packages
      DIS6  $NAME.dis  dis
      IC6  $NAME.ic  ic
      NPF6  $NAME.npf  npf
      WEL6  $NAME.wel  wel-1
      CHD6  $NAME.chd  chd_0
      HFB6  $NAME.hfb  hfb_0
      OC6  $NAME.oc  oc
    END packages
    """,
)
w(
    "$NAME.dis", """
    BEGIN options
      LENGTH_UNITS  feet
    END options

    BEGIN dimensions
      NLAY  $NLAY
      NROW  $NROW
      NCOL  $NCOL
    END dimensions

    BEGIN griddata
      delr
        CONSTANT  100.0
      delc
        CONSTANT  100.0
      top
        CONSTANT  100.0
      botm  LAYERED
        CONSTANT  50.0
        CONSTANT  0.0
    END griddata
    """,
)
w(
    "$NAME.ic", """
    BEGIN griddata
      strt
        CONSTANT  75.0
    END griddata
    """,
)
w(
    "$NAME.npf", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      icelltype
        CONSTANT  1
      k
        CONSTANT  10.0
    END griddata
    """,
)
w(
    "$NAME.wel", """
    BEGIN options
      PRINT_INPUT
      PRINT_FLOWS
    END options

    BEGIN dimensions
      MAXBOUND  1
    END dimensions

    BEGIN period  1
      2  6  6  -1000.0
    END period  1
    """,
)
w(
    "$NAME.chd", """
    BEGIN dimensions
      MAXBOUND  $(2NROW)
    END dimensions

    BEGIN period  1
    $(join(("  1  $i  1  100.0" for i in 1:NROW), "\n"))
    $(join(("  1  $i  $NCOL  75.0" for i in 1:NROW), "\n"))
    END period  1
    """,
)
w(
    "$NAME.hfb", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXHFB  $(NROW * NCOL)
    END dimensions

    BEGIN period  1
    $(join(("  1  $i  $j  2  $i  $j  2.0e-5" for i in 1:NROW for j in 1:NCOL), "\n"))
    END period  1
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
