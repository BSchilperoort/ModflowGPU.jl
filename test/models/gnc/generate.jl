# Writes the GNC6 test models of T6.13 into test/models/gnc/<case>/ (run with mf6 at test
# time, group gnc). MF6 has no GNC autotest in this checkout (GNC is tested through the
# external test models), so these are built here:
#
# - gnc_disv: a nested grid in one model, DISV. A 7 × 7 grid of 90 m cells whose centre
#   block (rows 3–5, columns 3–5) is refined 3 × 3 into 30 m cells, 2 confined layers (the
#   coarse cells along the block carry the hanging vertices of the fine cells, so DISV
#   connects them). Cells numbered by row of their centres (north to south, then west to
#   east), so the ghost-node cell is the lower-numbered one of some records and the higher
#   of others. One GNC record per horizontal coarse–fine connection: the ghost node in the
#   coarse cell n opposite the centre of the fine cell m, interpolated between n and its
#   coarse neighbour j along the interface (α_j = offset / 90 m; a centred fine cell has
#   no contributing cell, cellidsj 0). CHD (west and east columns of layer 1), WEL (two
#   fine cells, layer 2), RCH (layer 1), GHB (a coarse cell of layer 2); a steady-state
#   period, then a transient one of 3 steps (SS 1e-5). Implicit GNC (the default);
#   EXPLICIT is a variant made at test time.
# - gnc_disu: the same grid and stresses as DISU (connections from the cell rectangles:
#   CL12 the distances to the shared face, HWVA its length, ANGLDEGX).
# - gnc_lgr: the same refinement as two models (like ex-gwf-lgrv-lgr): parent DIS 2 × 7 × 7
#   (90 m, IDOMAIN 0 under the child), child DIS 2 × 9 × 9 (30 m, XORIGIN/YORIGIN 180 m) and
#   a GWF6-GWF6 exchange with the parent–child connections (CL1 45, CL2 15, HWVA 30) and
#   GNC6_FILENAME: one record per exchange connection, in the same order (ghost node in the
#   parent cell, contributing parent neighbour).
#
#   julia test/models/gnc/generate.jl
const ROOT = @__DIR__
const D = 90.0       # coarse cell size
const R = 3          # refinement
const DF = D / R     # fine cell size
const NC = 7         # coarse rows and columns
const RB = 3:5       # refined coarse rows and columns
const TOP = 0.0
const BOTM = (-10.0, -25.0)
const NLAY = 2

function write_case(case, files::Dict{String, String})
    dir = joinpath(ROOT, case)
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    for (f, t) in files
        write(joinpath(dir, f), t)
    end
    return dir
end

fmt(x::Real) = x == round(x) ? string(round(Int, x)) * ".0" : string(x)

# --- The nested grid of one layer: rectangles (x0, x1, y0, y1), coarse flag, coarse (row,
# column), sorted by centre (north to south, west to east).
struct Cell
    x0::Float64
    x1::Float64
    y0::Float64
    y1::Float64
    coarse::Bool
    r::Int
    c::Int
end
xc(c::Cell) = (c.x0 + c.x1) / 2
yc(c::Cell) = (c.y0 + c.y1) / 2
area(c::Cell) = (c.x1 - c.x0) * (c.y1 - c.y0)

function layer_cells()
    cells = Cell[]
    ytop = NC * D
    for r in 1:NC, c in 1:NC
        x0, y1 = (c - 1) * D, ytop - (r - 1) * D
        if r in RB && c in RB
            for fr in 1:R, fc in 1:R
                push!(cells, Cell(x0 + (fc - 1) * DF, x0 + fc * DF, y1 - fr * DF, y1 - (fr - 1) * DF, false, r, c))
            end
        else
            push!(cells, Cell(x0, x0 + D, y1 - D, y1, true, r, c))
        end
    end
    return sort(cells; by = c -> (-yc(c), xc(c)))
end

const CELLS = layer_cells()
const NCPL = length(CELLS)

# Horizontal neighbours in a layer: (i, j, shared length, angle from i to j in degrees,
# distance from i's centre to the face, from j's).
function horizontal_connections()
    out = Tuple{Int, Int, Float64, Float64, Float64, Float64}[]
    for i in 1:NCPL, j in 1:NCPL
        i == j && continue
        a, b = CELLS[i], CELLS[j]
        if a.x1 == b.x0 || a.x0 == b.x1          # shared vertical face
            ov = min(a.y1, b.y1) - max(a.y0, b.y0)
            ov > 0 || continue
            ang = a.x1 == b.x0 ? 0.0 : 180.0
            push!(out, (i, j, ov, ang, (a.x1 - a.x0) / 2, (b.x1 - b.x0) / 2))
        elseif a.y1 == b.y0 || a.y0 == b.y1      # shared horizontal face
            ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            ov > 0 || continue
            ang = a.y1 == b.y0 ? 90.0 : 270.0
            push!(out, (i, j, ov, ang, (a.y1 - a.y0) / 2, (b.y1 - b.y0) / 2))
        end
    end
    return out
end
const HCONN = horizontal_connections()

coarse_index(r, c) = findfirst(x -> x.coarse && x.r == r && x.c == c, CELLS)

# GNC records of one layer: (coarse cell n, fine cell m, contributing coarse cell j or 0, α).
function gnc_records()
    recs = Tuple{Int, Int, Int, Float64}[]
    for (i, j, _, ang, _, _) in HCONN
        a, b = CELLS[i], CELLS[j]
        (a.coarse && !b.coarse) || continue
        # Offset of the fine centre from the coarse centre along the interface.
        if ang == 0.0 || ang == 180.0
            t = yc(b) - yc(a)
            nb = t == 0 ? 0 : coarse_index(a.r + (t > 0 ? -1 : 1), a.c)
        else
            t = xc(b) - xc(a)
            nb = t == 0 ? 0 : coarse_index(a.r, a.c + (t > 0 ? 1 : -1))
        end
        nb = something(nb, 0)
        t != 0 && nb == 0 && error("no coarse neighbour for the ghost node of cell $i")
        push!(recs, (i, j, nb, abs(t) / D))
    end
    return recs
end

# --- Shared input files

const IMS = """
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
  LINEAR_ACCELERATION  BICGSTAB
  SCALING_METHOD  NONE
  REORDERING_METHOD  NONE
  RELAXATION_FACTOR  0.0
END linear
"""

const TDIS = """
BEGIN options
  TIME_UNITS  days
END options

BEGIN dimensions
  NPER  2
END dimensions

BEGIN perioddata
  1.0  1  1.0
  10.0  3  1.5
END perioddata
"""

mfsim(models, exchanges = String[]; name) = """
BEGIN options
END options

BEGIN timing
  TDIS6  $name.tdis
END timing

BEGIN models
$(join(("  gwf6  $m.nam  $m" for m in models), "\n"))
END models

BEGIN exchanges
$(join(("  " * e for e in exchanges), "\n"))
END exchanges

BEGIN solutiongroup  1
  ims6  $name.ims  $(join(models, "  "))
END solutiongroup  1
"""

nam(pkgs) = """
BEGIN options
  SAVE_FLOWS
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

# K 5 / 2 and K33 0.5 / 0.2 in layers 1 / 2 (`disu`: arrays by node, LAYERED is not
# available on DISU).
function npf(disu = false)
    arr(v1, v2) = disu ? "    INTERNAL  FACTOR  1.0\n      " * join([fill(v1, NCPL); fill(v2, NCPL)], " ") : "  LAYERED\n    CONSTANT  $v1\n    CONSTANT  $v2"
    return """
    BEGIN options
      SAVE_FLOWS
    END options

    BEGIN griddata
      icelltype
        CONSTANT  0
      k$(disu ? "\n" : "")$(arr(5.0, 2.0))
      k33$(disu ? "\n" : "")$(arr(0.5, 0.2))
    END griddata
    """
end

const STO = """
BEGIN options
  SAVE_FLOWS
END options

BEGIN griddata
  iconvert
    CONSTANT  0
  ss
    CONSTANT  1.0e-5
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

ic(strt) = "BEGIN options\nEND options\n\nBEGIN griddata\n  strt\n    CONSTANT  $strt\nEND griddata\n"

listpkg(lines; maxbound = length(lines)) = """
BEGIN options
  SAVE_FLOWS
END options

BEGIN dimensions
  MAXBOUND  $maxbound
END dimensions

BEGIN period  1
$(join(("  " * l for l in lines), "\n"))
END period  1
"""

function gncfile(lines; explicit = false)
    return """
    BEGIN options
    $(explicit ? "  EXPLICIT\n" : "")END options

    BEGIN dimensions
      NUMGNC  $(length(lines))
      NUMALPHAJ  1
    END dimensions

    BEGIN gncdata
    $(join(("  " * l for l in lines), "\n"))
    END gncdata
    """
end

# --- One model with the nested grid (DISV or DISU)

# Stresses on the nested grid, as (layer, cell in layer) → strings by a cellid function.
west_east = [i for i in 1:NCPL if CELLS[i].coarse && CELLS[i].c in (1, NC)]
const FINE = [i for i in 1:NCPL if !CELLS[i].coarse]
fine_wells = FINE[[20, 62]]
ghb_cell = coarse_index(6, 4)

function stress_files(name, cid)
    chd = [cid(1, i) * "  " * (CELLS[i].c == 1 ? "10.0" : "4.0") for i in west_east]
    wel = [cid(2, fine_wells[1]) * "  -150.0", cid(2, fine_wells[2]) * "  40.0"]
    rch = [cid(1, i) * "  1.0e-3" for i in 1:NCPL]
    ghb = [cid(2, ghb_cell) * "  6.0  25.0"]
    return Dict(
        "$name.chd" => listpkg(chd), "$name.wel" => listpkg(wel), "$name.rch" => listpkg(rch), "$name.ghb" => listpkg(ghb),
    )
end

function single_model(case, grid)
    name = case
    node(k, i) = (k - 1) * NCPL + i
    cid = grid === :disv ? ((k, i) -> "$k  $i") : ((k, i) -> string(node(k, i)))
    zero = grid === :disv ? "0  0" : "0"
    recs = gnc_records()
    gnc = String[]
    for k in 1:NLAY, (n, m, j, a) in recs
        push!(gnc, "$(cid(k, n))  $(cid(k, m))  $(j == 0 ? zero : cid(k, j))  $(fmt(a))")
    end
    files = Dict(
        "mfsim.nam" => mfsim([name]; name), "$name.tdis" => TDIS, "$name.ims" => IMS,
        "$name.nam" => nam(
            [
                "$(uppercase(String(grid)))6  $name.$grid  dis", "NPF6  $name.npf  npf", "IC6  $name.ic  ic", "STO6  $name.sto  sto",
                "GNC6  $name.gnc  gnc", "CHD6  $name.chd  chd", "WEL6  $name.wel  wel", "RCH6  $name.rch  rch", "GHB6  $name.ghb  ghb",
                "OC6  $name.oc  oc",
            ],
        ),
        "$name.npf" => npf(grid === :disu), "$name.sto" => STO, "$name.ic" => ic(7.0), "$name.oc" => oc(name), "$name.gnc" => gncfile(gnc),
        "$name.$grid" => grid === :disv ? disv_file() : disu_file(),
    )
    merge!(files, stress_files(name, cid))
    return write_case(case, files)
end

function disv_file()
    # Vertices: every corner of every cell (hanging ones included), on a 30 m lattice.
    verts = sort!(unique!([(x, y) for c in CELLS for x in (c.x0, c.x1) for y in (c.y0, c.y1)]); by = v -> (-v[2], v[1]))
    vid = Dict(v => i for (i, v) in enumerate(verts))
    cell2d = String[]
    for (i, c) in enumerate(CELLS)
        on(v) = (c.x0 <= v[1] <= c.x1 && c.y0 <= v[2] <= c.y1) && (v[1] in (c.x0, c.x1) || v[2] in (c.y0, c.y1))
        bv = [v for v in verts if on(v)]
        # Clockwise from the north-west corner: north edge west→east, east edge north→south,
        # south edge east→west, west edge south→north.
        key(v) = v[2] == c.y1 && v[1] < c.x1 ? (1, v[1]) : v[1] == c.x1 && v[2] > c.y0 ? (2, -v[2]) :
            v[2] == c.y0 && v[1] > c.x0 ? (3, -v[1]) : (4, v[2])
        sort!(bv; by = key)
        push!(cell2d, "$i  $(fmt(xc(c)))  $(fmt(yc(c)))  $(length(bv))  " * join((vid[v] for v in bv), "  "))
    end
    return """
    BEGIN options
      LENGTH_UNITS  meters
    END options

    BEGIN dimensions
      NLAY  $NLAY
      NCPL  $NCPL
      NVERT  $(length(verts))
    END dimensions

    BEGIN griddata
      top
        CONSTANT  $TOP
      botm  LAYERED
    $(join(("    CONSTANT  $b" for b in BOTM), "\n"))
    END griddata

    BEGIN vertices
    $(join(("  $i  $(fmt(v[1]))  $(fmt(v[2]))" for (i, v) in enumerate(verts)), "\n"))
    END vertices

    BEGIN cell2d
    $(join(("  " * l for l in cell2d), "\n"))
    END cell2d
    """
end

function disu_file()
    nodes = NLAY * NCPL
    tops = [k == 1 ? TOP : BOTM[k - 1] for k in 1:NLAY]
    rows = [Tuple{Int, Int, Float64, Float64, Float64}[] for _ in 1:nodes]  # (m, ihc, cl12, hwva, angldegx)
    for k in 1:NLAY
        th = tops[k] - BOTM[k]
        for (i, j, len, ang, cli, _) in HCONN
            push!(rows[(k - 1) * NCPL + i], ((k - 1) * NCPL + j, 1, cli, len, ang))
        end
        for i in 1:NCPL
            n = (k - 1) * NCPL + i
            k > 1 && push!(rows[n], (n - NCPL, 0, th / 2, area(CELLS[i]), 0.0))
            k < NLAY && push!(rows[n], (n + NCPL, 0, th / 2, area(CELLS[i]), 0.0))
        end
    end
    foreach(r -> sort!(r; by = first), rows)
    line(f, n) = join([f(nothing); [f(c) for c in rows[n]]], " ")
    iac = [1 + length(r) for r in rows]
    ja = [line(c -> isnothing(c) ? string(n) : string(c[1]), n) for n in 1:nodes]
    ihc = [line(c -> isnothing(c) ? "1" : string(c[2]), n) for n in 1:nodes]
    cl12 = [line(c -> isnothing(c) ? "0.0" : fmt(c[3]), n) for n in 1:nodes]
    hwva = [line(c -> isnothing(c) ? "0.0" : fmt(c[4]), n) for n in 1:nodes]
    ang = [line(c -> isnothing(c) ? "0.0" : fmt(c[5]), n) for n in 1:nodes]
    top = [tops[k] for k in 1:NLAY for _ in 1:NCPL]
    bot = [BOTM[k] for k in 1:NLAY for _ in 1:NCPL]
    ar = [area(CELLS[i]) for _ in 1:NLAY for i in 1:NCPL]
    blk(v) = join(("      " * l for l in v), "\n")
    return """
    BEGIN options
      LENGTH_UNITS  meters
    END options

    BEGIN dimensions
      NODES  $nodes
      NJA  $(sum(iac))
    END dimensions

    BEGIN griddata
      top
        INTERNAL  FACTOR  1.0
    $(blk([join(fmt.(top), " ")]))
      bot
        INTERNAL  FACTOR  1.0
    $(blk([join(fmt.(bot), " ")]))
      area
        INTERNAL  FACTOR  1.0
    $(blk([join(fmt.(ar), " ")]))
    END griddata

    BEGIN connectiondata
      iac
        INTERNAL  FACTOR  1
    $(blk([join(string.(iac), " ")]))
      ja
        INTERNAL  FACTOR  1
    $(blk(ja))
      ihc
        INTERNAL  FACTOR  1
    $(blk(ihc))
      cl12
        INTERNAL  FACTOR  1.0
    $(blk(cl12))
      hwva
        INTERNAL  FACTOR  1.0
    $(blk(hwva))
      angldegx
        INTERNAL  FACTOR  1.0
    $(blk(ang))
    END connectiondata
    """
end

# --- Parent and child models with GNC on the exchange

function lgr_model()
    case = "gnc_lgr"
    name = "gnclgr"
    np = NC
    nf = R * length(RB)
    xo = (first(RB) - 1) * D
    yo = (NC - last(RB)) * D
    # Parent IDOMAIN: 0 under the child.
    idom = [(r in RB && c in RB) ? 0 : 1 for r in 1:np, c in 1:np]
    idomtxt = join((join(string.(idom[r, :]), " ") for r in 1:np), "\n      ")
    disfile(n, delr, xorigin, yorigin; idomain = nothing) = """
    BEGIN options
      LENGTH_UNITS  meters
      XORIGIN  $xorigin
      YORIGIN  $yorigin
    END options

    BEGIN dimensions
      NLAY  $NLAY
      NROW  $n
      NCOL  $n
    END dimensions

    BEGIN griddata
      delr
        CONSTANT  $delr
      delc
        CONSTANT  $delr
      top
        CONSTANT  $TOP
      botm  LAYERED
    $(join(("    CONSTANT  $b" for b in BOTM), "\n"))
    $(isnothing(idomain) ? "" : "  idomain  LAYERED\n" * join(("    INTERNAL  FACTOR  1\n      $idomain" for _ in 1:NLAY), "\n"))
    END griddata
    """
    # Exchange connections and GNC records: parent cell (k, r, c) next to the child, child
    # edge cell (k, fr, fc), contributing parent cell along the interface.
    exg = String[]
    gnc = String[]
    r1, r2 = first(RB), last(RB)
    for k in 1:NLAY
        for r in 1:np, c in 1:np
            idom[r, c] == 1 || continue
            sides = Tuple{Int, Int, Int, Int}[]  # (child row, child col, parent nb row, nb col) per fine cell
            if r in RB && c == r1 - 1        # west of the child
                for f in 1:R
                    push!(sides, ((r - r1) * R + f, 1, r, c))
                end
            elseif r in RB && c == r2 + 1    # east
                for f in 1:R
                    push!(sides, ((r - r1) * R + f, nf, r, c))
                end
            elseif c in RB && r == r1 - 1    # north
                for f in 1:R
                    push!(sides, (1, (c - r1) * R + f, r, c))
                end
            elseif c in RB && r == r2 + 1    # south
                for f in 1:R
                    push!(sides, (nf, (c - r1) * R + f, r, c))
                end
            end
            for (fr, fc, _, _) in sides
                ns = c in (r1 - 1, r2 + 1) && r in RB   # west/east: interface along y (rows)
                f = ns ? fr - (r - r1) * R : fc - (c - r1) * R   # 1..R along the interface
                off = f - (R + 1) / 2                            # fine offset in fine cells, from the parent centre
                jr, jc = ns ? (r + (off > 0 ? 1 : -1), c) : (r, c + (off > 0 ? 1 : -1))
                a = abs(off) * DF / D
                push!(exg, "$k $r $c  $k $fr $fc  1  $(D / 2)  $(DF / 2)  $DF")
                push!(gnc, "$k $r $c  $k $fr $fc  " * (off == 0 ? "0 0 0" : "$k $jr $jc") * "  $(fmt(a))")
            end
        end
    end
    exgfile = """
    BEGIN options
      PRINT_FLOWS
      SAVE_FLOWS
      GNC6  FILEIN  $name.gnc
    END options

    BEGIN dimensions
      NEXG  $(length(exg))
    END dimensions

    BEGIN exchangedata
    $(join(("  " * e for e in exg), "\n"))
    END exchangedata
    """
    # Stresses: CHD on the parent's west and east columns (layer 1), WEL in the child,
    # RCH on both layer 1, GHB in the parent.
    chd = ["1 $r 1  10.0" for r in 1:np]
    append!(chd, ["1 $r $np  4.0" for r in 1:np])
    pr = ["1 $r $c  1.0e-3" for r in 1:np, c in 1:np if idom[r, c] == 1]
    cr = ["1 $r $c  1.0e-3" for r in 1:nf, c in 1:nf]
    files = Dict(
        "mfsim.nam" => mfsim(["parent", "child"], ["GWF6-GWF6  $name.gwfgwf  parent  child"]; name),
        "$name.tdis" => TDIS, "$name.ims" => IMS, "$name.gwfgwf" => exgfile, "$name.gnc" => gncfile(gnc),
        "parent.nam" => nam(
            [
                "DIS6  parent.dis  dis", "NPF6  parent.npf  npf", "IC6  parent.ic  ic", "STO6  parent.sto  sto",
                "CHD6  parent.chd  chd", "RCH6  parent.rch  rch", "GHB6  parent.ghb  ghb", "OC6  parent.oc  oc",
            ],
        ),
        "child.nam" => nam(
            [
                "DIS6  child.dis  dis", "NPF6  child.npf  npf", "IC6  child.ic  ic", "STO6  child.sto  sto",
                "WEL6  child.wel  wel", "RCH6  child.rch  rch", "OC6  child.oc  oc",
            ],
        ),
        "parent.dis" => disfile(np, D, 0.0, 0.0; idomain = idomtxt), "child.dis" => disfile(nf, DF, xo, yo),
        "parent.npf" => npf(), "child.npf" => npf(), "parent.sto" => STO, "child.sto" => STO,
        "parent.ic" => ic(7.0), "child.ic" => ic(7.0), "parent.oc" => oc("parent"), "child.oc" => oc("child"),
        "parent.chd" => listpkg(chd), "parent.rch" => listpkg(pr), "parent.ghb" => listpkg(["2 6 4  6.0  25.0"]),
        "child.rch" => listpkg(cr), "child.wel" => listpkg(["2 2 8  -150.0", "2 7 3  40.0"]),
    )
    return write_case(case, files)
end

single_model("gnc_disv", :disv)
single_model("gnc_disu", :disu)
lgr_model()
