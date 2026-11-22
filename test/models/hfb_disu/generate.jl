# Writes test/models/hfb_disu (T6.11), the model of MF6's autotest
# test_gwf_disu_hfb_idomain.py: DISU 3 × 3 × 3 (as flopy's `get_disu_kwargs`, cells 10 × 10,
# layers 10 thick) with user node 2 excluded (IDOMAIN 0), CHD at user nodes 1 and 27, and
# one barrier between user nodes 26 and 27 (user numbers above the active-cell count).
# Steady state, confined. Run with mf6 at test time (group hfb).
#
#   julia test/models/hfb_disu/generate.jl
const DIR = @__DIR__
const NAME = "disuhfb"
const NLAY, NROW, NCOL = 3, 3, 3
const D = 10.0

w(name, txt) = write(joinpath(DIR, name), txt)

node(k, i, j) = (k - 1) * NROW * NCOL + (i - 1) * NCOL + j
nodes = NLAY * NROW * NCOL
# Connections in ascending node order: (m, ihc, cl12, hwva).
rows = [Tuple{Int, Int, Float64, Float64}[] for _ in 1:nodes]
for k in 1:NLAY, i in 1:NROW, j in 1:NCOL
    n = node(k, i, j)
    for (dk, di, dj) in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0))
        kk, ii, jj = k + dk, i + di, j + dj
        (1 <= kk <= NLAY && 1 <= ii <= NROW && 1 <= jj <= NCOL) || continue
        vert = dk != 0
        push!(rows[n], (node(kk, ii, jj), vert ? 0 : 1, D / 2, vert ? D * D : D))
    end
end
iac = [1 + length(r) for r in rows]
ja = [join([string(n); [string(c[1]) for c in rows[n]]], " ") for n in 1:nodes]
ihc = [join(["1"; [string(c[2]) for c in rows[n]]], " ") for n in 1:nodes]
cl12 = [join(["0.0"; [string(c[3]) for c in rows[n]]], " ") for n in 1:nodes]
hwva = [join(["0.0"; [string(c[4]) for c in rows[n]]], " ") for n in 1:nodes]
top = [-(k - 1) * D for k in 1:NLAY for _ in 1:(NROW * NCOL)]
bot = [-k * D for k in 1:NLAY for _ in 1:(NROW * NCOL)]
idomain = [n == 2 ? 0 : 1 for n in 1:nodes]
nja = sum(iac)

block(lines) = join(("    " * l for l in lines), "\n")

w(
    "mfsim.nam", """
    # MF6 autotest test_gwf_disu_hfb_idomain.py (T6.11): DISU with IDOMAIN, HFB on user node numbers.
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
      1.0  1  1.0
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
      OUTER_MAXIMUM  100
    END nonlinear

    BEGIN linear
      INNER_MAXIMUM  300
      INNER_DVCLOSE  1.0e-10
      INNER_RCLOSE  1.0e-10
      LINEAR_ACCELERATION  CG
    END linear
    """,
)
w(
    "$NAME.nam", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN packages
      DISU6  $NAME.disu  disu
      IC6  $NAME.ic  ic
      NPF6  $NAME.npf  npf
      CHD6  $NAME.chd  chd_0
      OC6  $NAME.oc  oc
      HFB6  $NAME.hfb  hfb_0
    END packages
    """,
)
w(
    "$NAME.disu", """
    BEGIN options
    END options

    BEGIN dimensions
      NODES  $nodes
      NJA  $nja
    END dimensions

    BEGIN griddata
      top
        INTERNAL  FACTOR  1.0
    $(block([join(string.(top), " ")]))
      bot
        INTERNAL  FACTOR  1.0
    $(block([join(string.(bot), " ")]))
      area
        CONSTANT  $(D * D)
      idomain
        INTERNAL  FACTOR  1
    $(block([join(string.(idomain), " ")]))
    END griddata

    BEGIN connectiondata
      iac
        INTERNAL  FACTOR  1
    $(block([join(string.(iac), " ")]))
      ja
        INTERNAL  FACTOR  1
    $(block(ja))
      ihc
        INTERNAL  FACTOR  1
    $(block(ihc))
      cl12
        INTERNAL  FACTOR  1.0
    $(block(cl12))
      hwva
        INTERNAL  FACTOR  1.0
    $(block(hwva))
    END connectiondata
    """,
)
w(
    "$NAME.ic", """
    BEGIN griddata
      strt
        CONSTANT  0.0
    END griddata
    """,
)
w(
    "$NAME.npf", """
    BEGIN options
    END options

    BEGIN griddata
      icelltype
        CONSTANT  0
      k
        CONSTANT  1.0
    END griddata
    """,
)
w(
    "$NAME.chd", """
    BEGIN dimensions
      MAXBOUND  2
    END dimensions

    BEGIN period  1
      1  1.0
      $nodes  0.0
    END period  1
    """,
)
w(
    "$NAME.hfb", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXHFB  1
    END dimensions

    BEGIN period  1
      26  27  1.0e-6
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
