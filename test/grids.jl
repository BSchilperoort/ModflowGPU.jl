# Synthetic grids and comparison helpers shared by the connectivity and NPF tests.
using ModflowGPU: DisSpec, DisvSpec, DisuSpec

"A DIS spec with the given layer bottoms `(ncol, nrow, nlay)` and IDOMAIN."
function _dis(; delr, delc, top, botm, idomain = ones(Int32, size(botm)))
    ncol, nrow, nlay = size(botm)
    return DisSpec(;
        nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr = Float64.(delr), delc = Float64.(delc),
        top = Float64.(top), botm = Float64.(botm), idomain = Int32.(idomain), file = "test.dis",
    )
end

# A small deterministic generator (no Random dependency): values in 0:(m - 1).
mutable struct _Lcg
    s::UInt64
end
_next!(g::_Lcg, m) = (g.s = g.s * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Int((g.s >> 33) % m))

# Elementwise relative closeness of two arrays of equal length.
_rel_close(a, b; rtol = 1.0e-14) = length(a) == length(b) && all(abs(x - y) <= rtol * max(abs(x), abs(y)) for (x, y) in zip(a, b))

# A coupling-like grid (PLAN.md section 1.1): 2 layers, variable DELR/DELC, sloping top,
# IDOMAIN 0 outside the domain, 1 in layer 1 and 2 in layer 2, with columns where only one
# layer is active.
function _synthetic_coupling_dis()
    ncol, nrow, nlay = 7, 5, 2
    delr = [1000.0, 950.5, 900.25, 1000.0, 1100.0, 1000.0, 875.125]
    delc = [925.0, 1000.0, 1000.0, 1050.75, 990.0]
    top = [250.0 + 3.25 * j - 7.5 * i + 0.125 * i * j for j in 1:ncol, i in 1:nrow]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    botm[:, :, 1] = top .- [20.0 + 0.5 * j + i for j in 1:ncol, i in 1:nrow]
    botm[:, :, 2] = botm[:, :, 1] .- [150.0 - 2.0 * j + 3.0 * i for j in 1:ncol, i in 1:nrow]
    idomain = zeros(Int32, ncol, nrow, nlay)
    for i in 1:nrow, j in 1:ncol
        inside = !(i == 1 && j >= 5) && !(j == 1 && i >= 4) && !(i == nrow && j == ncol)
        idomain[j, i, 1] = inside && (i, j) != (3, 4) ? 1 : 0   # a hole in layer 1
        idomain[j, i, 2] = inside && (i, j) != (2, 2) ? 2 : 0   # and one in layer 2
    end
    idomain[5, 1, 2] = 2  # layer 2 active below an excluded layer-1 cell
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end

# A 4-layer grid with vertical pass-through cells (IDOMAIN -1): single and stacked
# pass-through cells, pass-through over and under excluded cells, in the top and bottom
# layers, and next to active cells (no horizontal connections to pass-through cells).
function _synthetic_passthrough_dis()
    ncol, nrow, nlay = 4, 3, 4
    delr = [10.0, 12.5, 7.25, 10.0]
    delc = [8.0, 11.0, 9.5]
    top = fill(100.0, ncol, nrow) .+ [0.5 * j - 0.25 * i for j in 1:ncol, i in 1:nrow]
    thick = [5.0, 7.5, 3.25, 12.0]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    for k in 1:nlay
        botm[:, :, k] = (k == 1 ? top : botm[:, :, k - 1]) .- (thick[k] .+ [0.1 * j * k + 0.05 * i for j in 1:ncol, i in 1:nrow])
    end
    # columns (j, i) => IDOMAIN by layer
    cols = Dict(
        (1, 1) => (1, -1, 1, 1), (2, 1) => (1, -1, -1, 1), (3, 1) => (-1, 1, 1, -1), (4, 1) => (1, -1, 0, 1),
        (1, 2) => (1, 0, -1, 1), (2, 2) => (-1, -1, -1, 1), (3, 2) => (1, 1, 1, 1), (4, 2) => (1, -1, -1, -1),
        (1, 3) => (0, -1, 1, 1), (2, 3) => (1, 1, -1, 0), (3, 3) => (-1, 0, 1, -1), (4, 3) => (1, 2, -3, 3),
    )
    idomain = Array{Int32}(undef, ncol, nrow, nlay)
    for ((j, i), ids) in cols, k in 1:nlay
        idomain[j, i, k] = ids[k]
    end
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end

# The DISV description of a DIS grid (T6.4): vertices at the cell corners, numbered row by
# row from the top-left corner, cells clockwise from their top-left corner, cell2d number
# `(i - 1) * ncol + j`, so user node numbers (and IDOMAIN, TOP, BOTM) are those of the DIS
# grid. Coordinates as Dis.f90 computes the cell centres (row 1 at the top).
function _dis_as_disv(dis::DisSpec)
    ncol, nrow, nlay = Int(dis.ncol), Int(dis.nrow), Int(dis.nlay)
    xv = [0.0; cumsum(dis.delr)]
    yv = reverse([0.0; cumsum(reverse(dis.delc))])  # yv[1] is the top edge
    vid(i, j) = Int32((i - 1) * (ncol + 1) + j)       # corner i in 1:nrow+1, j in 1:ncol+1
    vertices = Matrix{Float64}(undef, 2, (nrow + 1) * (ncol + 1))
    for i in 1:(nrow + 1), j in 1:(ncol + 1)
        vertices[:, vid(i, j)] = [xv[j], yv[i]]
    end
    ncpl = nrow * ncol
    cell_center = Matrix{Float64}(undef, 2, ncpl)
    cell_vertices = Vector{Vector{Int32}}(undef, ncpl)
    for i in 1:nrow, j in 1:ncol
        c = (i - 1) * ncol + j
        cell_center[:, c] = [xv[j] + 0.5 * dis.delr[j], yv[i + 1] + 0.5 * dis.delc[i]]
        cell_vertices[c] = [vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)]
    end
    return DisvSpec(;
        nlay = Int32(nlay), ncpl = Int32(ncpl), nvert = Int32(size(vertices, 2)), top = vec(dis.top),
        botm = reshape(dis.botm, ncpl, nlay), idomain = reshape(dis.idomain, ncpl, nlay), vertices, cell_center,
        cell_vertices, file = "test.disv",
    )
end

# The DISU description of a DIS grid without pass-through cells (T6.4): the connections of
# the grid with every cell active (from the DIS connectivity, so the same CL12, HWVA and
# ANGLDEGX), with the grid's IDOMAIN (2 → 1) given to DISU. User node numbers are those of
# the DIS grid. `vertices = true` adds the DISV vertices and cell centres.
function _dis_as_disu(dis::DisSpec; vertices::Bool = false, angldegx::Bool = true)
    all(>=(0), dis.idomain) || throw(ArgumentError("DISU has no pass-through cells"))
    full = DisSpec(; nlay = dis.nlay, nrow = dis.nrow, ncol = dis.ncol, delr = dis.delr, delc = dis.delc, top = dis.top, botm = dis.botm)
    c = ModflowGPU.build_connectivity(full)
    nodes = Int(c.nodes)
    iac = Int32[c.ia[n + 1] - c.ia[n] for n in 1:nodes]
    ihc = zeros(Int32, c.nja)
    cl12 = zeros(c.nja)
    hwva = zeros(c.nja)
    ang = zeros(c.nja)
    for n in 1:nodes, ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
        s = c.jas[ii]
        m = c.ja[ii]
        ihc[ii] = c.ihc[s]
        cl12[ii] = m > n ? c.cl1[s] : c.cl2[s]
        hwva[ii] = c.hwva[s]
        a = rad2deg(c.anglex[s])
        ang[ii] = m > n ? a : mod(a + 180.0, 360.0)
    end
    disv = vertices ? _dis_as_disv(dis) : nothing
    return DisuSpec(;
        nodes = Int32(nodes), nja = c.nja, nvert = vertices ? disv.nvert : Int32(0), top = c.top, bot = c.bot, area = c.area,
        idomain = Int32[min(d, 1) for d in vec(dis.idomain)], iac, ja = copy(c.ja), ihc, cl12, hwva,
        angldegx = angldegx ? ang : nothing,
        vertices = vertices ? disv.vertices : zeros(2, 0),
        cell_center = vertices ? repeat(disv.cell_center, 1, Int(dis.nlay)) : zeros(2, 0),
        cell_vertices = vertices ? repeat(disv.cell_vertices, Int(dis.nlay)) : Vector{Int32}[], file = "test.disu",
    )
end
