# Generates test/models/coupling_rde (T2.4 synthetic coupling model): julia generate.jl <outdir>
out = ARGS[1]
mkpath(out)
w(name, s) = write(joinpath(out, name), s)
nlay, nrow, ncol = 2, 12, 15
delr = [100.0 + 10.0 * ((j * 7) % 5) for j in 1:ncol]
delc = [100.0 + 12.5 * ((i * 3) % 4) for i in 1:nrow]
top = [22.0 + 0.9 * (j - 1) + 0.15 * (i - 1) + 0.25 * sin(i + 2j) for i in 1:nrow, j in 1:ncol]
bot1 = top .- 10.0
bot2 = bot1 .- 30.0
idom = ones(Int, nlay, nrow, ncol)
for i in 1:nrow, j in 1:ncol
    if (i <= 2 && j >= 13) || (i >= 11 && j <= 2) || (i == 12 && j == 15)
        idom[:, i, j] .= 0
    end
end
idom[1, 6, 9] = 0   # a hole in layer 1
fmt(x) = string(round(x; digits = 4))
arr(a) = join([join(fmt.(a[i, :]), " ") for i in 1:size(a, 1)], "\n      ")
iarr(a) = join([join(a[i, :], " ") for i in 1:size(a, 1)], "\n      ")
active(k, i, j) = idom[k, i, j] > 0

hdr = "# Synthetic coupling-like model for the RIV/DRN/EVT residual test (T2.4), generated\n# by a script; confined, 2 layers, 12 x 15 cells.\n"
w(
    "mfsim.nam", hdr * """
        BEGIN options
        END options

        BEGIN timing
          TDIS6  rde.tdis
        END timing

        BEGIN models
          gwf6  rde.nam  rde
        END models

        BEGIN exchanges
        END exchanges

        BEGIN solutiongroup  1
          ims6  rde.ims  rde
        END solutiongroup  1
        """
)
w(
    "rde.tdis", """
    BEGIN options
      TIME_UNITS  days
    END options

    BEGIN dimensions
      NPER  3
    END dimensions

    BEGIN perioddata
      1.0  1  1.0
      30.0  5  1.2
      20.0  4  1.0
    END perioddata
    """
)
w(
    "rde.ims", """
    BEGIN options
      PRINT_OPTION  SUMMARY
    END options

    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-10
      OUTER_MAXIMUM  200
    END nonlinear

    BEGIN linear
      INNER_MAXIMUM  1000
      INNER_DVCLOSE  1.0e-13
      INNER_RCLOSE  1.0e-10
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """
)
w(
    "rde.nam", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN packages
      DIS6  rde.dis  dis
      NPF6  rde.npf  npf
      IC6  rde.ic  ic
      STO6  rde.sto  sto
      CHD6  rde.chd  chd
      WEL6  rde.wel  wel
      RCH6  rde.rcha  rcha
      RIV6  rde.riv  riv
      DRN6  rde.drn  drn
      EVT6  rde.evt  evt
      EVT6  rde.evta  evta
      OC6  rde.oc  oc
    END packages
    """
)
w(
    "rde.dis", """
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
        INTERNAL
          $(join(fmt.(delr), " "))
      delc
        INTERNAL
          $(join(fmt.(delc), " "))
      top
        INTERNAL
          $(arr(top))
      botm  LAYERED
        INTERNAL
          $(arr(bot1))
        INTERNAL
          $(arr(bot2))
      idomain  LAYERED
        INTERNAL
          $(iarr(idom[1, :, :]))
        INTERNAL
          $(iarr(idom[2, :, :]))
    END griddata
    """
)
w(
    "rde.npf", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      icelltype
        CONSTANT  0
      k  LAYERED
        CONSTANT  12.0
        CONSTANT  4.0
      k33  LAYERED
        CONSTANT  1.5
        CONSTANT  0.5
    END griddata
    """
)
w(
    "rde.ic", """
    BEGIN griddata
      strt
        CONSTANT  23.0
    END griddata
    """
)
w(
    "rde.sto", """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      iconvert
        CONSTANT  0
      ss
        CONSTANT  2.0e-5
      sy
        CONSTANT  0.1
    END griddata

    BEGIN period  1
      STEADY-STATE
    END period  1

    BEGIN period  2
      TRANSIENT
    END period  2
    """
)
w(
    "rde.oc", """
    BEGIN options
      BUDGET  FILEOUT  rde.cbc
      HEAD  FILEOUT  rde.hds
    END options

    BEGIN period  1
      SAVE  HEAD  ALL
      SAVE  BUDGET  ALL
    END period  1
    """
)
# CHD: west edge, layer 1.
chd = ["  1 $i 1  $(fmt(21.0 + 0.05 * i))" for i in 1:nrow if active(1, i, 1)]
w(
    "rde.chd", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXBOUND  $(length(chd))
    END dimensions

    BEGIN period  1
    $(join(chd, "\n"))
    END period  1
    """
)
w(
    "rde.wel", """
    BEGIN options
    END options

    BEGIN dimensions
      MAXBOUND  2
    END dimensions

    BEGIN period  2
      2 8 11  -900.0
      2 4 6  -300.0
    END period  2

    BEGIN period  3
      2 8 11  -1500.0
    END period  3
    """
)
w(
    "rde.rcha", """
    BEGIN options
      READASARRAYS
    END options

    BEGIN period  1
      recharge
        CONSTANT  0.0012
    END period  1

    BEGIN period  3
      recharge
        CONSTANT  0.0004
    END period  3
    """
)
# RIV: a river along column 6..8 meandering north-south in layer 1, stage 1 m below top,
# rbot 2.5 m below top; a MULTIPLIER on the conductance.
riv = String[]
for i in 1:nrow
    j = 6 + (i % 3)
    active(1, i, j) || continue
    push!(riv, "  1 $i $j  $(fmt(top[i, j] - 1.0 - 0.3 * (i % 2)))  $(fmt(150.0 + 10i))  $(fmt(top[i, j] - 2.5))  $(i % 4 == 0 ? 0.5 : 1.0)")
end
riv3 = [
    replace(
        l, r"  (\S+)  (\S+)  (\S+)  (\S+)$" => s -> begin
            v = split(strip(s))
            "  $(fmt(parse(Float64, v[1]) + 0.6))  $(v[2])  $(v[3])  $(v[4])"
        end
    ) for l in riv
]
w(
    "rde.riv", """
    BEGIN options
      AUXILIARY  cmult
      AUXMULTNAME  cmult
      BOUNDNAMES
    END options

    BEGIN dimensions
      MAXBOUND  $(length(riv))
    END dimensions

    BEGIN period  1
    $(join([l * "  river" for l in riv], "\n"))
    END period  1

    BEGIN period  3
    $(join([l * "  river" for l in riv3], "\n"))
    END period  3
    """
)
# DRN: layer 1, columns 3..5 and 10..12 of some rows, elevation 0.6 m below top, with a
# drainage depth (AUXDEPTHNAME) of 0, +1.5 or -1.0 m.
drn = String[]
for i in 2:2:nrow, j in (3, 4, 5, 10, 11, 12)
    active(1, i, j) || continue
    dd = (i + j) % 3 == 0 ? 0.0 : ((i + j) % 3 == 1 ? 1.5 : -1.0)
    push!(drn, "  1 $i $j  $(fmt(top[i, j] - 0.6))  $(fmt(80.0 + 5j))  $(fmt(dd))")
end
w(
    "rde.drn", """
    BEGIN options
      AUXILIARY  ddrn
      AUXDEPTHNAME  ddrn
    END options

    BEGIN dimensions
      MAXBOUND  $(length(drn))
    END dimensions

    BEGIN period  1
    $(join(drn, "\n"))
    END period  1
    """
)
# EVT list, NSEG 3, SURF_RATE_SPECIFIED: odd rows of layer 1, surface at top, extinction
# depth 3 m (rows < 7) or 1.5 m.
evt = String[]
for i in 1:2:nrow, j in 1:ncol
    active(1, i, j) || continue
    j in (6, 7, 8) && continue
    x = i < 7 ? 3.0 : 1.5
    push!(evt, "  1 $i $j  $(fmt(top[i, j]))  $(fmt(0.002 + 0.0001 * j))  $(fmt(x))  0.25 0.6  0.65 0.3  0.9")
end
w(
    "rde.evt", """
    BEGIN options
      SURF_RATE_SPECIFIED
    END options

    BEGIN dimensions
      MAXBOUND  $(length(evt))
      NSEG  3
    END dimensions

    BEGIN period  1
    $(join(evt, "\n"))
    END period  1

    BEGIN period  3
    $(join([replace(l, "0.25 0.6  0.65 0.3  0.9" => "0.25 0.6  0.65 0.3  1.0") for l in evt], "\n"))
    END period  3
    """
)
# EVTA: the even rows, surface 0.3 m below top, extinction depth 2 m, layer 2 under the
# layer-1 hole.
ievt = ones(Int, nrow, ncol)
ievt[6, 9] = 2
rate = [iseven(i) ? 0.0015 : 0.0 for i in 1:nrow, j in 1:ncol]
w(
    "rde.evta", """
    BEGIN options
      READASARRAYS
    END options

    BEGIN period  1
      ievt
        INTERNAL
          $(iarr(ievt))
      surface
        INTERNAL
          $(arr(top .- 0.3))
      rate
        INTERNAL
          $(arr(rate))
      depth
        CONSTANT  2.0
    END period  1
    """
)
