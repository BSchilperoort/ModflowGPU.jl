# Host-side grid connectivity in MF6's layout (PLAN.md section 3.3), shared by DIS (dis.jl),
# DISV (disv.jl) and DISU (disu.jl, T6.4).
#
# Ports the grid-independent parts of MF6 6.8.2 Connections.f90 (`fillisym`, `filljas`,
# `iajausr`, `getjaindex`) and of the `grid_finalize` routines of Dis.f90/Disv.f90
# (`nodereduced`/`nodeuser`).
#
# Storage (same as MF6's `ConnectionsType`, so tests compare with libmf6's `<MODEL>/CON/*`
# arrays element by element):
# - CSR over reduced nodes: `ia` `(nodes + 1)`, `ja` `(nja)`, with the diagonal stored
#   first in each row and the off-diagonals in increasing node order (the order MF6 adds
#   them in, which is sorted for DIS and DISV).
# - `isym` `(nja)`: position of the transposed entry (`isym[ia[n]] = ia[n]`).
# - `jas` `(nja)`: symmetric connection number in `1:njas`, 0 on the diagonal. The upper
#   triangle (`ja[ii] > n`) is numbered in CSR order and the lower triangle takes the
#   number of its transposed entry.
# - Geometry per symmetric connection (`njas`), indexed by `jas`: `ihc`, `cl1`, `cl2`,
#   `hwva`, `anglex`. `cl1` belongs to the lower-numbered node of the pair (n < m), `cl2`
#   to the higher one, as in MF6: for entry `ii` of row `n`, the distance from `n` is
#   `cl1[jas[ii]]` when `ja[ii] > n` and `cl2[jas[ii]]` otherwise (`connection_lengths`).
#   Per-symmetric-connection storage halves memory and keeps `condsat` (T2.2) symmetric by
#   construction.

"""
    Connectivity

Host-side discretization of a GWF model on reduced (active) nodes, in MF6's layout
(Connections.f90 `ConnectionsType` plus the reduced cell arrays of the DIS packages).
Built by `build_connectivity`. Geometry is `Float64`, indices `Int32`.

Nodes:
- `nodes`: number of reduced nodes (cells with IDOMAIN > 0); `nodesuser`: number of user
  nodes (all cells).
- `nodeuser` `(nodes)`: user node of each reduced node.
- `nodereduced` `(nodesuser)`: reduced node of each user node, `0` for an excluded cell
  (IDOMAIN = 0) and `-1` for a vertical pass-through cell (IDOMAIN < 0), as in MF6.
  Unlike MF6, both maps are always filled (MF6 keeps dummies of length 1 when no cell is
  excluded).
- `top`, `bot`, `area` `(nodes)`: cell top, bottom and plan area; `xc`, `yc` `(nodes)`:
  cell centre in model coordinates (no origin or rotation applied); empty for a DISU grid
  without vertices (as in MF6).
- `grid`: `:dis`, `:disv` or `:disu`, the discretization the connectivity was built from.

Connections (CSR, diagonal first in each row; see the header of
src/discretization/connectivity.jl):
- `nja`, `njas = (nja - nodes) ÷ 2`; `ia` `(nodes + 1)`, `ja`, `isym`, `jas` `(nja)`.
- `ihc` `(njas)`: 0 for a vertical connection, 1 for a horizontal one (2 for a
  horizontal connection between vertically staggered DISU cells).
- `cl1`, `cl2` `(njas)`: distance from the lower-numbered (`cl1`) and higher-numbered
  (`cl2`) node to the shared face; `hwva` `(njas)`: face width for horizontal
  connections, face area for vertical ones; `anglex` `(njas)`: angle (radians) between
  the x axis and the connection normal pointing from the lower- to the higher-numbered
  node (horizontal connections); `DNODATA` (3e30) for a DISU grid without ANGLDEGX.
"""
struct Connectivity
    nodes::Int32
    nodesuser::Int32
    nja::Int32
    njas::Int32
    nodeuser::Vector{Int32}
    nodereduced::Vector{Int32}
    ia::Vector{Int32}
    ja::Vector{Int32}
    isym::Vector{Int32}
    jas::Vector{Int32}
    ihc::Vector{Int32}
    cl1::Vector{Float64}
    cl2::Vector{Float64}
    hwva::Vector{Float64}
    anglex::Vector{Float64}
    top::Vector{Float64}
    bot::Vector{Float64}
    area::Vector{Float64}
    xc::Vector{Float64}
    yc::Vector{Float64}
    grid::Symbol
end

function Base.show(io::IO, c::Connectivity)
    return print(io, "Connectivity(nodes = ", c.nodes, " of ", c.nodesuser, ", nja = ", c.nja, ", njas = ", c.njas, ")")
end

"""
    reduced_numbering(idomain) -> (nodereduced, nodeuser)

Reduced node numbers from IDOMAIN by user node (Dis.f90 `grid_finalize`): cells with
IDOMAIN > 0 are numbered in user node order; `nodereduced` is `0` for IDOMAIN = 0
(excluded) and `-1` for IDOMAIN < 0 (vertical pass-through). `nodeuser` maps reduced to
user nodes.
"""
function reduced_numbering(idomain::AbstractVector{<:Integer})
    nodesuser = length(idomain)
    nodesuser <= typemax(Int32) || throw(ArgumentError("more than typemax(Int32) user nodes"))
    nodereduced = Vector{Int32}(undef, nodesuser)
    nodes = count(>(0), idomain)
    nodeuser = Vector{Int32}(undef, nodes)
    noder = Int32(0)
    for (n, id) in enumerate(idomain)
        if id > 0
            noder += Int32(1)
            nodereduced[n] = noder
            nodeuser[noder] = n
        else
            nodereduced[n] = id < 0 ? Int32(-1) : Int32(0)
        end
    end
    return nodereduced, nodeuser
end

"""
    fill_isym!(isym, ia, ja) -> isym

Position of the transposed entry of every CSR entry (Connections.f90 `fillisym`); 0 when
the transposed entry does not exist, the entry itself on the diagonal.
"""
function fill_isym!(isym::AbstractVector{Int32}, ia::AbstractVector{Int32}, ja::AbstractVector{Int32})
    for n in 1:(length(ia) - 1)
        for ii in ia[n]:(ia[n + 1] - 1)
            m = ja[ii]
            if m == n
                isym[ii] = ii
            else
                isym[ii] = 0
                for jj in ia[m]:(ia[m + 1] - 1)
                    if ja[jj] == n
                        isym[ii] = jj
                        break
                    end
                end
            end
        end
    end
    return isym
end

"""
    fill_jas!(jas, ia, ja, isym) -> njas

Symmetric connection numbers (Connections.f90 `filljas`): the upper triangle is numbered
in CSR order, the lower triangle takes the number of the transposed entry, and the
diagonal is 0. Returns the number of symmetric connections.
"""
function fill_jas!(jas::AbstractVector{Int32}, ia::AbstractVector{Int32}, ja::AbstractVector{Int32}, isym::AbstractVector{Int32})
    neq = length(ia) - 1
    ipos = Int32(0)
    for n in 1:neq
        jas[ia[n]] = 0
        for ii in (ia[n] + 1):(ia[n + 1] - 1)
            if ja[ii] > n
                ipos += Int32(1)
                jas[ii] = ipos
            end
        end
    end
    for n in 1:neq, ii in ia[n]:(ia[n + 1] - 1)
        ja[ii] < n && (jas[ii] = jas[isym[ii]])
    end
    return ipos
end

"""
    iajausr(c::Connectivity) -> (iausr, jausr)

The CSR structure in user node numbers, as written to the binary grid file (Connections.f90
`iajausr`): `iausr` `(nodesuser + 1)` has empty rows for cells outside the solution, and
`jausr` `(nja)` is `ja` with user node numbers.
"""
function iajausr(c::Connectivity)
    iausr = Vector{Int32}(undef, c.nodesuser + 1)
    iausr[end] = c.ia[end]
    for n in c.nodesuser:-1:1
        nr = c.nodereduced[n]
        iausr[n] = nr < 1 ? iausr[n + 1] : c.ia[nr]
    end
    jausr = Int32[c.nodeuser[m] for m in c.ja]
    return iausr, jausr
end

"""
    getjaindex(c::Connectivity, n, m) -> Int

Position in `ja` of the connection from reduced node `n` to `m` (Connections.f90
`getjaindex`): `ia[n]` when `n == m`, 0 when the nodes are not connected and -1 when a
node number is out of range.
"""
function getjaindex(c::Connectivity, n::Integer, m::Integer)
    (1 <= n <= c.nodes && 1 <= m <= c.nodes) || return -1
    n == m && return Int(c.ia[n])
    for ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
        c.ja[ii] == m && return Int(ii)
    end
    return 0
end

"""
    connection_lengths(c::Connectivity, n, ii) -> (cln, clm)

Distances from node `n` and from its neighbour `m = ja[ii]` (off-diagonal entry `ii` of
row `n`) to the shared face: `(cl1, cl2)` of the symmetric connection when `n < m`,
`(cl2, cl1)` otherwise (as gwf-npf.f90 swaps them, e.g. in `calc_spdis`).
"""
function connection_lengths(c::Connectivity, n::Integer, ii::Integer)
    isympos = c.jas[ii]
    return c.ja[ii] > n ? (c.cl1[isympos], c.cl2[isympos]) : (c.cl2[isympos], c.cl1[isympos])
end

"""
    build_connectivity(dis::AbstractDisSpec) -> Connectivity
    build_connectivity(model::GwfSpec) -> Connectivity

Reduced nodes, cell geometry and the CSR connectivity of a grid, after MF6's setup-time
grid checks (which throw an `MF6InputError` with MF6's messages): DIS
(src/discretization/dis.jl), DISV (disv.jl) and DISU (disu.jl).
"""
build_connectivity(model::GwfSpec) = build_connectivity(model.dis)
