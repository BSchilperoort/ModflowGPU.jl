# Grid layout of the binary output files: how MF6 splits an array over the user nodes
# into records (`record_array` of Dis.f90, Disv.f90, Disu.f90) and which grid dimensions
# it writes in the budget-file headers (`record_array`, `record_srcdst_list_header`).

"""
    OutputLayout(kind, ncol, nrow, nlay)
    OutputLayout(dis::AbstractDisSpec)
    OutputLayout(grb::BinaryGridFile)

Grid layout of the arrays in head and budget files, in MF6's `mshape` terms:

- `:dis`: `(ncol, nrow, nlay)`; head files have one record per layer of `ncol × nrow`;
- `:disv`: `(ncpl, 1, nlay)`; one head record per layer of `ncpl` values;
- `:disu`: `(nodes, 1, 1)`; one head record of `nodes` values.

`(ncol, nrow, nlay)` are also the NDIM1-3 of the budget-file headers. Arrays written with
a layout are vectors (or arrays) of length [`nodesuser`](@ref)`(layout)` in user-node
order.
"""
struct OutputLayout
    kind::Symbol
    ncol::Int
    nrow::Int
    nlay::Int
    function OutputLayout(kind::Symbol, ncol::Integer, nrow::Integer, nlay::Integer)
        kind in (:dis, :disv, :disu) || throw(ArgumentError("layout kind must be :dis, :disv or :disu, got :$kind"))
        (ncol >= 0 && nrow >= 1 && nlay >= 1) || throw(ArgumentError("invalid layout dimensions ($ncol, $nrow, $nlay)"))
        kind === :dis || nrow == 1 || throw(ArgumentError("a :$kind layout has nrow = 1, got $nrow"))
        kind === :disu && nlay != 1 && throw(ArgumentError("a :disu layout has nlay = 1, got $nlay"))
        return new(kind, ncol, nrow, nlay)
    end
end

OutputLayout(d::DisSpec) = OutputLayout(:dis, d.ncol, d.nrow, d.nlay)
OutputLayout(d::DisvSpec) = OutputLayout(:disv, d.ncpl, 1, d.nlay)
OutputLayout(d::DisuSpec) = OutputLayout(:disu, d.nodes, 1, 1)

function OutputLayout(g::BinaryGridFile)
    g.grid == "DIS" && return OutputLayout(:dis, g["NCOL"], g["NROW"], g["NLAY"])
    g.grid == "DISV" && return OutputLayout(:disv, g["NCPL"], 1, g["NLAY"])
    g.grid == "DISU" && return OutputLayout(:disu, g["NODES"], 1, 1)
    throw(ArgumentError("unknown grid type $(g.grid) in binary grid file"))
end

"""
    OutputLayout(values::AbstractArray)

Layout from the shape of an array in MF6 memory shape: 3-D `(ncol, nrow, nlay)` is DIS,
2-D `(ncpl, nlay)` is DISV, a vector is DISU. A convenience only: a flat DIS or DISV
vector is taken as DISU, so pass an explicit layout for those.
"""
function OutputLayout(values::AbstractArray)
    N = ndims(values)
    N == 3 && return OutputLayout(:dis, size(values)...)
    N == 2 && return OutputLayout(:disv, size(values, 1), 1, size(values, 2))
    N == 1 && return OutputLayout(:disu, length(values), 1, 1)
    throw(ArgumentError("grid arrays have 1 (DISU), 2 (DISV) or 3 (DIS) dimensions, got $N"))
end

"""
    nodesuser(layout::OutputLayout) -> Int

Number of user nodes (`ncol * nrow * nlay`).
"""
nodesuser(l::OutputLayout) = l.ncol * l.nrow * l.nlay

"""
    budget_dims(layout::OutputLayout) -> NTuple{3, Int}

The NDIM1-3 that MF6 writes in budget-file headers for this grid (NDIM3 is stored
negative in the file).
"""
budget_dims(l::OutputLayout) = (l.ncol, l.nrow, l.nlay)

function _check_length(layout::OutputLayout, n::Integer, what)
    n == nodesuser(layout) ||
        throw(DimensionMismatch("$what has $n values, the $(layout.kind) layout $(budget_dims(layout)) has $(nodesuser(layout)) user nodes"))
    return nothing
end
