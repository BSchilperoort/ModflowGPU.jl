# Vertex grid (DISV) → Connectivity (host only, T6.4).
#
# Ports MF6 6.8.2:
# - Disv.f90 `source_dimensions` (NLAY/NCPL/NVERT checks), `define_cellverts` (the vertex
#   list of each cell, closed by repeating the first vertex when the input does not),
#   `grid_finalize` (active node count, "no active nodes" and cell thickness checks,
#   `nodereduced`/`nodeuser`, reduced `top`, `bot`, `xc`, `yc`), `connect` and
#   `get_cell2d_area` (cell areas by the shoelace formula, with the area checks);
# - Connections.f90 `disvconnections` and `vertexconnect`: per row the diagonal first, then
#   the vertical neighbours (skipping pass-through cells, stopping at excluded cells, as in
#   DIS) and the cells of the same layer that share an edge, in increasing node order (MF6
#   builds a sparse matrix and sorts it);
# - DisvGeom.f90 `cprops`, `shared_edge`, `get_area`, `anglex`, `distance`,
#   `distance_normal`: the symmetric arrays `ihc`, `cl1`, `cl2`, `hwva`, `anglex` of every
#   connection `n < m`. Vertical: `hwva` the cell area, `cl1`/`cl2` half the thicknesses,
#   anglex 0. Horizontal: the shared edge (the first pair of consecutive vertices of cell n
#   that appears reversed in cell m), `hwva` its length, `cl1`/`cl2` the normal distances
#   of the cell centres to it, anglex the angle of its outward normal from cell n.
#
# Rows are built per node from a per-cell2d list of edge neighbours, so the work is
# O(nodes · neighbours) without a sparse matrix.

# Closed vertex lists of the cells in CSR form (Disv.f90 `define_cellverts`): the vertices
# as given, plus the first vertex again when the last one differs from it.
function _cell_vertex_csr(cell_vertices::Vector{Vector{Int32}})
    ncell = length(cell_vertices)
    iavert = Vector{Int32}(undef, ncell + 1)
    iavert[1] = 1
    for j in 1:ncell
        iv = cell_vertices[j]
        nv = length(iv) + (!isempty(iv) && length(iv) > 1 && iv[end] != iv[1] ? 1 : 0)
        iavert[j + 1] = iavert[j] + nv
    end
    javert = Vector{Int32}(undef, iavert[end] - 1)
    for j in 1:ncell
        iv = cell_vertices[j]
        p = iavert[j]
        for v in iv
            javert[p] = v
            p += 1
        end
        p < iavert[j + 1] && (javert[p] = iv[1])
    end
    return iavert, javert
end

# The signed shoelace sum of a cell (Disv.f90 `get_cell2d_area`, DisvGeom.f90 `get_area`,
# before the factor): area = -sum/2 for clockwise vertices.
function _shoelace_sum(vertices::AbstractMatrix{Float64}, iavert, javert, j::Integer)
    lo, hi = Int(iavert[j]), Int(iavert[j + 1]) - 1
    nvert = hi - lo + 1
    area = 0.0
    nvert < 1 && return area
    iv1 = javert[lo]
    x1 = vertices[1, iv1]
    y1 = vertices[2, iv1]
    icount = 1
    for ivert in lo:hi
        x = vertices[1, javert[ivert]]
        y = icount < nvert ? vertices[2, javert[ivert + 1]] : vertices[2, javert[lo]]
        area = area + (x - x1) * (y - y1)
        icount += 1
    end
    icount = 1
    for ivert in lo:hi
        y = vertices[2, javert[ivert]]
        x = icount < nvert ? vertices[1, javert[ivert + 1]] : vertices[1, javert[lo]]
        area = area - (x - x1) * (y - y1)
        icount += 1
    end
    return area
end

# DisvGeom.f90 `shared_edge`: the first pair of consecutive vertices (a, b) of list 1 such
# that list 2 has b followed by a (a clockwise face of one cell is counter-clockwise in the
# other); (0, 0) when the cells share no edge.
function _shared_edge(javert, lo1::Integer, hi1::Integer, lo2::Integer, hi2::Integer)
    for il1 in lo1:(hi1 - 1)
        a, b = javert[il1], javert[il1 + 1]
        for il2 in hi2:-1:(lo2 + 1)
            if a == javert[il2] && b == javert[il2 - 1]
                return a, b
            end
        end
    end
    return Int32(0), Int32(0)
end

_disv_distance(x1, y1, x2, y2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

# DisvGeom.f90 `distance_normal`: distance from (x0, y0) to the line through the two points.
_disv_distance_normal(x0, y0, x1, y1, x2, y2) =
    abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / _disv_distance(x1, y1, x2, y2)

"""
    disv_anglex(x1, y1, x2, y2) -> Float64

Angle (radians, in `[0, 2π)`) between the x axis and the normal of the edge from vertex
`(x1, y1)` to `(x2, y2)` that points out of a cell whose vertices are clockwise
(DisvGeom.f90 `anglex`): `atan2(x2 − x1, −(y2 − y1))`, plus 2π when negative.
"""
function disv_anglex(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
    dx = x2 - x1
    dy = y2 - y1
    ax = atan(dx, -dy)
    ax < 0.0 && (ax = 2.0 * Float64(pi) + ax)
    return ax
end

"""
    check_disv_grid(dis::DisvSpec) -> Vector{Float64}

MF6's setup-time checks of a DISV grid (Disv.f90 `source_dimensions`, `grid_finalize`,
`connect`): NLAY, NCPL, NVERT ≥ 1, at least one cell with IDOMAIN > 0, a positive
thickness of every cell with IDOMAIN > 0, and a positive area of every cell2d (clockwise
vertices). Throws an `MF6InputError` (for `dis.file`) with MF6's messages. Array shapes
that do not match the dimensions are an `ArgumentError`. Returns the cell2d areas.
"""
function check_disv_grid(dis::DisvSpec)
    file = isempty(dis.file) ? "DISV" : dis.file
    errors = String[]
    for (name, v) in (("NLAY", dis.nlay), ("NCPL", dis.ncpl), ("NVERT", dis.nvert))
        v < 1 && push!(errors, "$name was not specified or was specified incorrectly.")
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    ncpl, nlay, nvert = Int(dis.ncpl), Int(dis.nlay), Int(dis.nvert)
    size(dis.top) == (ncpl,) || throw(ArgumentError("DisvSpec: size(top) = $(size(dis.top)), expected ($ncpl,)"))
    for (name, a) in (("botm", dis.botm), ("idomain", dis.idomain))
        size(a) == (ncpl, nlay) || throw(ArgumentError("DisvSpec: size($name) = $(size(a)), expected ($ncpl, $nlay)"))
    end
    size(dis.vertices) == (2, nvert) || throw(ArgumentError("DisvSpec: size(vertices) = $(size(dis.vertices)), expected (2, $nvert)"))
    size(dis.cell_center) == (2, ncpl) || throw(ArgumentError("DisvSpec: size(cell_center) = $(size(dis.cell_center)), expected (2, $ncpl)"))
    length(dis.cell_vertices) == ncpl || throw(ArgumentError("DisvSpec: length(cell_vertices) = $(length(dis.cell_vertices)), expected $ncpl"))
    for (j, iv) in enumerate(dis.cell_vertices), v in iv
        1 <= v <= nvert || throw(ArgumentError("DisvSpec: vertex $v of cell $j is not in 1:$nvert"))
    end
    if !any(>(0), dis.idomain)
        throw(MF6InputError(file, 0, "Model does not have any active nodes. Ensure IDOMAIN array has some values greater than zero."))
    end
    for k in 1:nlay, j in 1:ncpl
        dis.idomain[j, k] > 0 || continue
        top = k > 1 ? dis.botm[j, k - 1] : dis.top[j]
        bot = dis.botm[j, k]
        if top - bot <= 0.0
            push!(errors, "CELL ($k,$j) THICKNESS <= 0. TOP, BOT: " * _fortran_1pg(top) * _fortran_1pg(bot))
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    # Disv.f90 `connect`: cell areas.
    iavert, javert = _cell_vertex_csr(dis.cell_vertices)
    area2d = Vector{Float64}(undef, ncpl)
    nlt = neq = 0
    for j in 1:ncpl
        a = -1.0 * _shoelace_sum(dis.vertices, iavert, javert, j) * 0.5
        area2d[j] = a
        if a < 0.0
            nlt += 1
            push!(errors, "Calculated CELL2D area less than zero for cell $j.")
        elseif a == 0.0
            neq += 1
            push!(errors, "Calculated CELL2D area is zero for cell $j.")
        end
    end
    nlt > 0 && push!(
        errors, "$nlt cell(s) have an area less than zero. Calculated cell areas must be greater than zero. " *
            "Negative areas often mean vertices are not listed in clockwise order.",
    )
    neq > 0 && push!(
        errors, "$neq cell(s) have an area equal to zero. Calculated cell areas must be greater than zero. " *
            "Calculated cell areas equal to zero indicate that the cell is not defined by a valid polygon.",
    )
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    return area2d
end

# Edge neighbours of every cell2d (Connections.f90 `vertexconnect`: cells that share a
# vertex and, by `shares_edge`, an edge), as a CSR `(iah, jah)` sorted by cell number.
function _disv_edge_neighbours(ncpl::Integer, nvert::Integer, iavert, javert)
    # Cells of every vertex.
    cnt = zeros(Int, nvert + 1)
    for j in 1:ncpl, p in iavert[j]:(iavert[j + 1] - 1)
        cnt[javert[p] + 1] += 1
    end
    iavc = cumsum(cnt) .+ 1
    javc = Vector{Int32}(undef, iavc[end] - 1)
    pos = copy(iavc)
    for j in 1:ncpl, p in iavert[j]:(iavert[j + 1] - 1)
        v = javert[p]
        javc[pos[v]] = j
        pos[v] += 1
    end
    iah = Vector{Int}(undef, ncpl + 1)
    iah[1] = 1
    jah = Int32[]
    buf = Int32[]
    for j in 1:ncpl
        empty!(buf)
        lo1, hi1 = iavert[j], iavert[j + 1] - 1
        for p in lo1:hi1, q in iavc[javert[p]]:(iavc[javert[p] + 1] - 1)
            j2 = javc[q]
            (j2 == j || j2 in buf) && continue
            a, _ = _shared_edge(javert, lo1, hi1, iavert[j2], iavert[j2 + 1] - 1)
            a != 0 && push!(buf, j2)
        end
        sort!(buf)
        append!(jah, buf)
        iah[j + 1] = length(jah) + 1
    end
    return iah, jah
end

# Reduced node of the first cell above (`step = -1`) or below (`step = 1`) cell (j, k) that
# is not a pass-through cell (Connections.f90 `vertexconnect`, "Up/Down direction").
@inline function _disv_vertical_neighbour(nr2::AbstractMatrix{Int32}, j, k, step)
    nlay = size(nr2, 2)
    kk = k + step
    while 1 <= kk <= nlay
        mr = nr2[j, kk]
        mr >= 0 && return mr
        kk += step
    end
    return Int32(0)
end

"""
    build_connectivity(dis::DisvSpec) -> Connectivity

The reduced nodes, cell geometry and connectivity of a vertex grid, as MF6's Disv.f90
`grid_finalize` and Connections.f90 `disvconnections` build them, after the grid checks
of `check_disv_grid`.
"""
function build_connectivity(dis::DisvSpec)
    area2d = check_disv_grid(dis)
    ncpl, nlay, nvert = Int(dis.ncpl), Int(dis.nlay), Int(dis.nvert)
    vert = dis.vertices
    cxy = dis.cell_center
    iavert, javert = _cell_vertex_csr(dis.cell_vertices)
    nodereduced, nodeuser = reduced_numbering(vec(dis.idomain))
    nodes = length(nodeuser)
    nodesuser = length(nodereduced)
    nr2 = reshape(nodereduced, ncpl, nlay)

    # Reduced cell arrays (Disv.f90 `grid_finalize`, `connect`).
    top = Vector{Float64}(undef, nodes)
    bot = Vector{Float64}(undef, nodes)
    area = Vector{Float64}(undef, nodes)
    xc = Vector{Float64}(undef, nodes)
    yc = Vector{Float64}(undef, nodes)
    for k in 1:nlay, j in 1:ncpl
        nr = nr2[j, k]
        nr <= 0 && continue
        top[nr] = k > 1 ? dis.botm[j, k - 1] : dis.top[j]
        bot[nr] = dis.botm[j, k]
        area[nr] = area2d[j]
        xc[nr] = cxy[1, j]
        yc[nr] = cxy[2, j]
    end

    # Rows: diagonal, then the sorted neighbours (vertical and same-layer edge neighbours).
    iah, jah = _disv_edge_neighbours(ncpl, nvert, iavert, javert)
    ia = Vector{Int32}(undef, nodes + 1)
    ia[1] = 1
    ja = Int32[]
    sizehint!(ja, nodes + 2 * length(jah) + 2 * nodes)
    buf = Int32[]
    for k in 1:nlay, j in 1:ncpl
        nr = nr2[j, k]
        nr <= 0 && continue
        empty!(buf)
        k > 1 && (mr = _disv_vertical_neighbour(nr2, j, k, -1)) > 0 && push!(buf, mr)
        k < nlay && (mr = _disv_vertical_neighbour(nr2, j, k, 1)) > 0 && push!(buf, mr)
        for p in iah[j]:(iah[j + 1] - 1)
            mr = nr2[jah[p], k]
            mr > 0 && push!(buf, mr)
        end
        sort!(buf)
        push!(ja, nr)
        append!(ja, buf)
        length(ja) < typemax(Int32) || throw(ArgumentError("more than typemax(Int32) - 1 matrix entries"))
        ia[nr + 1] = length(ja) + 1
    end
    nja = length(ja)
    isym = fill_isym!(Vector{Int32}(undef, nja), ia, ja)
    jas = Vector{Int32}(undef, nja)
    njas = Int(fill_jas!(jas, ia, ja, isym))

    # Symmetric arrays (DisvGeom.f90 `cprops` for every connection n < m).
    ihc = Vector{Int32}(undef, njas)
    cl1 = Vector{Float64}(undef, njas)
    cl2 = Vector{Float64}(undef, njas)
    hwva = Vector{Float64}(undef, njas)
    anglex = Vector{Float64}(undef, njas)
    for n in 1:nodes
        jn = (Int(nodeuser[n]) - 1) % ncpl + 1
        for ii in (ia[n] + 1):(ia[n + 1] - 1)
            m = ja[ii]
            m < n && continue
            s = jas[ii]
            jm = (Int(nodeuser[m]) - 1) % ncpl + 1
            if jn == jm
                ihc[s] = 0
                hwva[s] = abs(_shoelace_sum(vert, iavert, javert, jn)) * 0.5
                cl1[s] = 0.5 * (top[n] - bot[n])
                cl2[s] = 0.5 * (top[m] - bot[m])
                anglex[s] = 0.0
            else
                ihc[s] = 1
                iv1, iv2 = _shared_edge(javert, iavert[jn], iavert[jn + 1] - 1, iavert[jm], iavert[jm + 1] - 1)
                if iv1 == 0 || iv2 == 0
                    hwva[s] = 0.0
                    cl1[s] = 1.0
                    cl2[s] = 1.0
                    anglex[s] = 0.0  # left undefined by MF6 (never happens: rows hold edge neighbours only)
                else
                    x1, y1 = vert[1, iv1], vert[2, iv1]
                    x2, y2 = vert[1, iv2], vert[2, iv2]
                    hwva[s] = _disv_distance(x1, y1, x2, y2)
                    cl1[s] = _disv_distance_normal(cxy[1, jn], cxy[2, jn], x1, y1, x2, y2)
                    cl2[s] = _disv_distance_normal(cxy[1, jm], cxy[2, jm], x1, y1, x2, y2)
                    anglex[s] = disv_anglex(x1, y1, x2, y2)
                end
            end
        end
    end

    return Connectivity(
        Int32(nodes), Int32(nodesuser), Int32(nja), Int32(njas), nodeuser, nodereduced,
        ia, ja, isym, jas, ihc, cl1, cl2, hwva, anglex, top, bot, area, xc, yc, :disv,
    )
end
