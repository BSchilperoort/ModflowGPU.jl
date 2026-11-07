# Structured grid (DIS) → Connectivity (host only).
#
# Ports MF6 6.8.2:
# - Dis.f90 `source_dimensions` (NLAY/NROW/NCOL checks) and `grid_finalize`: active node
#   count, "no active nodes" and cell thickness checks, `nodereduced`/`nodeuser`,
#   `cellx`/`celly`, and the reduced `top`, `bot`, `area`, `xc`, `yc`. The top of a cell in
#   layer k > 1 is the bottom of layer k - 1 in the same column, whatever that cell's
#   IDOMAIN.
# - Connections.f90 `disconnections`: the CSR structure (rows in user node order; per row
#   the diagonal, then up, back (row i - 1), left, right, front (row i + 1), down) and the
#   symmetric arrays `ihc`, `cl1`, `cl2`, `hwva`, `anglex` (right, front and down
#   connections of each cell in user node order, which is the `jas` order). Vertical
#   connections skip pass-through cells (IDOMAIN < 0) and stop at excluded cells
#   (IDOMAIN = 0); `cl1`/`cl2` of a vertical connection are half the thickness of the two
#   connected cells, so the pass-through cells in between do not add to the distance.
#
# MF6 builds the rows with a `sparsematrix` and then `fillisym`/`filljas`; here the rows
# are built directly in two passes (count, fill), which gives the same arrays (the
# neighbours are added in increasing reduced node order) in O(nodes) time and without
# per-row allocations, for grids with 1e8 cells.

# Fortran `1PG24.15` (the number format of MF6's thickness error message).
function _fortran_1pg(x::Float64; w::Int = 24, d::Int = 15)
    a = abs(x)
    s = if a == 0
        @sprintf("%.*f", d - 1, x) * "    "
    else
        # G editing uses F when 0.1 <= |x| < 10^d after rounding to d significant digits.
        e = floor(Int, log10(parse(Float64, @sprintf("%.*E", d - 1, a)))) + 1
        # F editing: d significant digits; E editing with the 1P scale factor: d decimals.
        0 <= e <= d ? @sprintf("%#.*f", d - e, x) * "    " : @sprintf("%.*E", d, x)
    end
    return lpad(s, w)
end

"""
    check_dis_grid(dis::DisSpec)

MF6's setup-time checks of a DIS grid (Dis.f90 `source_dimensions`, `grid_finalize`):
NLAY, NROW, NCOL ≥ 1, at least one cell with IDOMAIN > 0, and a positive thickness for
every cell with IDOMAIN > 0. Throws an `MF6InputError` (for `dis.file`) with MF6's
messages, all thickness errors in one message. Array shapes that do not match the
dimensions are an `ArgumentError`.
"""
function check_dis_grid(dis::DisSpec)
    file = isempty(dis.file) ? "DIS" : dis.file
    errors = String[]
    for (name, v) in (("NLAY", dis.nlay), ("NROW", dis.nrow), ("NCOL", dis.ncol))
        v < 1 && push!(errors, "$name was not specified or was specified incorrectly.")
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    ncol, nrow, nlay = Int(dis.ncol), Int(dis.nrow), Int(dis.nlay)
    size(dis.delr) == (ncol,) || throw(ArgumentError("DisSpec: size(delr) = $(size(dis.delr)), expected ($ncol,)"))
    size(dis.delc) == (nrow,) || throw(ArgumentError("DisSpec: size(delc) = $(size(dis.delc)), expected ($nrow,)"))
    size(dis.top) == (ncol, nrow) || throw(ArgumentError("DisSpec: size(top) = $(size(dis.top)), expected ($ncol, $nrow)"))
    for (name, a) in (("botm", dis.botm), ("idomain", dis.idomain))
        size(a) == (ncol, nrow, nlay) || throw(ArgumentError("DisSpec: size($name) = $(size(a)), expected ($ncol, $nrow, $nlay)"))
    end
    if !any(>(0), dis.idomain)
        throw(
            MF6InputError(
                file, 0,
                "Model does not have any active nodes. Ensure IDOMAIN array has some values greater than zero.",
            ),
        )
    end
    for k in 1:nlay, i in 1:nrow, j in 1:ncol
        dis.idomain[j, i, k] < 1 && continue
        top = k > 1 ? dis.botm[j, i, k - 1] : dis.top[j, i]
        bot = dis.botm[j, i, k]
        if top - bot <= 0.0
            push!(errors, "CELL ($k,$i,$j) THICKNESS <= 0. TOP, BOT: " * _fortran_1pg(top) * _fortran_1pg(bot))
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    return nothing
end

# Reduced node of the first cell above (`step = -1`) or below (`step = 1`) cell (j, i, k)
# that is not a pass-through cell: > 0 for a connected cell, 0 for none (an excluded cell
# stops the search, as does the edge of the grid when only pass-through cells follow).
# Connections.f90 `disconnections`, "Up direction" / "Down direction".
@inline function _dis_vertical_neighbour(nr3::AbstractArray{Int32, 3}, j, i, k, step)
    nlay = size(nr3, 3)
    kk = k + step
    while 1 <= kk <= nlay
        mr = nr3[j, i, kk]
        mr >= 0 && return mr
        kk += step
    end
    return Int32(0)
end

"""
    build_connectivity(dis::DisSpec) -> Connectivity

The reduced nodes, cell geometry and connectivity of a structured grid, as MF6's
Dis.f90 `grid_finalize` and Connections.f90 `disconnections` build them, after the
grid checks of `check_dis_grid`.
"""
function build_connectivity(dis::DisSpec)
    check_dis_grid(dis)
    ncol, nrow, nlay = Int(dis.ncol), Int(dis.nrow), Int(dis.nlay)
    delr, delc = dis.delr, dis.delc
    nodereduced, nodeuser = reduced_numbering(vec(dis.idomain))
    nodes = length(nodeuser)
    nodesuser = length(nodereduced)
    nr3 = reshape(nodereduced, ncol, nrow, nlay)

    # Cell centres (Dis.f90 `grid_finalize`): row numbers increase in the -y direction.
    cellx = Vector{Float64}(undef, ncol)
    celly = Vector{Float64}(undef, nrow)
    cellx[1] = 0.5 * delr[1]
    for j in 2:ncol
        cellx[j] = cellx[j - 1] + 0.5 * delr[j - 1] + 0.5 * delr[j]
    end
    celly[nrow] = 0.5 * delc[nrow]
    for i in (nrow - 1):-1:1
        celly[i] = celly[i + 1] + 0.5 * delc[i + 1] + 0.5 * delc[i]
    end

    # Reduced cell arrays.
    top = Vector{Float64}(undef, nodes)
    bot = Vector{Float64}(undef, nodes)
    area = Vector{Float64}(undef, nodes)
    xc = Vector{Float64}(undef, nodes)
    yc = Vector{Float64}(undef, nodes)
    for k in 1:nlay, i in 1:nrow, j in 1:ncol
        nr = nr3[j, i, k]
        nr <= 0 && continue
        top[nr] = k > 1 ? dis.botm[j, i, k - 1] : dis.top[j, i]
        bot[nr] = dis.botm[j, i, k]
        area[nr] = delr[j] * delc[i]
        xc[nr] = cellx[j]
        yc[nr] = celly[i]
    end

    # Pass 1: entries per row (diagonal included) and the number of upper-triangle entries.
    ia = Vector{Int32}(undef, nodes + 1)
    ia[1] = 1
    njas = 0
    for k in 1:nlay, i in 1:nrow, j in 1:ncol
        nr = nr3[j, i, k]
        nr <= 0 && continue
        nlower = Int(k > 1 && _dis_vertical_neighbour(nr3, j, i, k, -1) > 0) +
            Int(i > 1 && nr3[j, i - 1, k] > 0) + Int(j > 1 && nr3[j - 1, i, k] > 0)
        nupper = Int(j < ncol && nr3[j + 1, i, k] > 0) + Int(i < nrow && nr3[j, i + 1, k] > 0) +
            Int(k < nlay && _dis_vertical_neighbour(nr3, j, i, k, 1) > 0)
        njas += nupper
        rowlen = ia[nr] + 1 + nlower + nupper
        rowlen <= typemax(Int32) || throw(ArgumentError("more than typemax(Int32) - 1 matrix entries"))
        ia[nr + 1] = rowlen
    end
    nja = Int(ia[nodes + 1]) - 1
    @assert nja == nodes + 2 * njas

    # Pass 2: `ja`, and the symmetric arrays in `jas` order (right, front, down per cell).
    ja = Vector{Int32}(undef, nja)
    jas = Vector{Int32}(undef, nja)
    ihc = Vector{Int32}(undef, njas)
    cl1 = Vector{Float64}(undef, njas)
    cl2 = Vector{Float64}(undef, njas)
    hwva = Vector{Float64}(undef, njas)
    anglex = Vector{Float64}(undef, njas)
    angle_front = 3.0 / 2.0 * Float64(pi)  # Connections.f90: DTHREE / DTWO * DPI
    isympos = Int32(0)
    for k in 1:nlay, i in 1:nrow, j in 1:ncol
        nr = nr3[j, i, k]
        nr <= 0 && continue
        ii = ia[nr]
        ja[ii] = nr
        jas[ii] = 0
        # Lower triangle: `jas` is filled below, from the transposed entry.
        for mr in (
                k > 1 ? _dis_vertical_neighbour(nr3, j, i, k, -1) : Int32(0),
                i > 1 ? nr3[j, i - 1, k] : Int32(0),
                j > 1 ? nr3[j - 1, i, k] : Int32(0),
            )
            mr > 0 || continue
            ii += Int32(1)
            ja[ii] = mr
        end
        # Upper triangle (Connections.f90 `disconnections`, "Fill symmetric
        # discretization arrays").
        if j < ncol && (mr = nr3[j + 1, i, k]) > 0  # right
            ii += Int32(1)
            isympos += Int32(1)
            ja[ii] = mr
            jas[ii] = isympos
            ihc[isympos] = 1
            cl1[isympos] = 0.5 * delr[j]
            cl2[isympos] = 0.5 * delr[j + 1]
            hwva[isympos] = delc[i]
            anglex[isympos] = 0.0
        end
        if i < nrow && (mr = nr3[j, i + 1, k]) > 0  # front
            ii += Int32(1)
            isympos += Int32(1)
            ja[ii] = mr
            jas[ii] = isympos
            ihc[isympos] = 1
            cl1[isympos] = 0.5 * delc[i]
            cl2[isympos] = 0.5 * delc[i + 1]
            hwva[isympos] = delr[j]
            anglex[isympos] = angle_front
        end
        if k < nlay && (mr = _dis_vertical_neighbour(nr3, j, i, k, 1)) > 0  # down
            ii += Int32(1)
            isympos += Int32(1)
            ja[ii] = mr
            jas[ii] = isympos
            ihc[isympos] = 0
            cl1[isympos] = 0.5 * (top[nr] - bot[nr])
            cl2[isympos] = 0.5 * (top[mr] - bot[mr])
            hwva[isympos] = delr[j] * delc[i]
            anglex[isympos] = 0.0
        end
        @assert ii == ia[nr + 1] - 1
    end

    # Transposed positions (Connections.f90 `fillisym`) and the lower half of `jas`
    # (`filljas`).
    isym = fill_isym!(Vector{Int32}(undef, nja), ia, ja)
    for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
        ja[ii] < n && (jas[ii] = jas[isym[ii]])
    end

    return Connectivity(
        Int32(nodes), Int32(nodesuser), Int32(nja), Int32(njas), nodeuser, nodereduced,
        ia, ja, isym, jas, ihc, cl1, cl2, hwva, anglex, top, bot, area, xc, yc,
    )
end
